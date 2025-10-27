from functools import partial
from typing import Optional

import datasets
import torch
from datasets import DatasetDict
from torch.distributed.nn.functional import all_gather  # PyTorch ≥ 2.1
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker

from colpali_engine.data.sampler import SingleDatasetBatchSampler

#Added 
from torch.nn import CrossEntropyLoss



def concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.cat(all_gather(t), dim=0)  # keeps grad graph
    return t


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], DatasetDict):
            dataset_list = list(kwargs["train_dataset"].values())
        elif isinstance(kwargs["train_dataset"], list):
            dataset_list = kwargs["train_dataset"]
        else:
            dataset_list = None

        if isinstance(dataset_list, list):
            # round down each dataset if not divible by global batch size
            batch_size = kwargs["args"].train_batch_size
            for i in range(len(dataset_list)):
                if len(dataset_list[i]) % batch_size != 0:
                    total_samples = (len(dataset_list[i]) // batch_size) * batch_size
                    dataset_list[i] = dataset_list[i].take(total_samples)

        if dataset_list is not None:
            kwargs["train_dataset"] = ConcatDataset(dataset_list)

        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.args.remove_unused_columns = False  # Safety, don't remove dataset columns from dataloader
        self.dataset_list = dataset_list

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataset = self.train_dataset
        description = "Training"
        batch_size = self._train_batch_size
        sampler_fn = self._get_train_sampler
        is_training = True
        dataloader_key = None

        if self.dataset_list is None:
            return super()._get_dataloader(dataset, description, batch_size, sampler_fn, is_training, dataloader_key)

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        dataloader_params = {
            ######### don't set batch size, mutually exclusive from batch sampler ######
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                ###### batch_sampler set instead of sampler in trainer code #######
                dataloader_params["batch_sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = DataLoader(dataset, **dataloader_params)

        # Accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version for eval dataloaders.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return self.accelerator.prepare(dataloader)

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if self.dataset_list is None:
            return super()._get_train_sampler(train_dataset=train_dataset)

        # Use SingleDatasetBatchSampler to ensure that each dataset in the list is sampled independently
        # Note: Surely breaks in distributed training
        # TODO: fix this
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )


        
    #Modified

    # Allow training with in-batch negatives and multiple hard negatives simultaniously
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_fct = CrossEntropyLoss()
        temperature = 0.02

        query_embs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        pos_doc_embs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc_")})

        # Case 1: Training with hard negs 
        if "neg_doc_input_ids" in inputs:
            neg_doc_embs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc_")})
            
            pos_doc_embs_global = concat_all_gather(pos_doc_embs)
            
            in_batch_scores = torch.einsum('bd,cd->bc', query_embs, pos_doc_embs_global)
            
            batch_size_local = query_embs.size(0)
            num_negatives_per_query = neg_doc_embs.size(0) // batch_size_local
            neg_doc_embs_reshaped = neg_doc_embs.view(batch_size_local, num_negatives_per_query, -1)
            hard_neg_scores = torch.einsum('bd,bkd->bk', query_embs, neg_doc_embs_reshaped)
            
            all_scores = torch.cat([in_batch_scores, hard_neg_scores], dim=1)
            
            offset = self.accelerator.process_index * batch_size_local
            labels = torch.arange(batch_size_local, device=all_scores.device) + offset
            
            loss = loss_fct(all_scores / temperature, labels)

            return (loss, (query_embs, pos_doc_embs)) if return_outputs else loss

        # Case 2: Evaluation (no hard neg)
        else:
            
            scores = torch.einsum("bd,cd->bc", query_embs, pos_doc_embs)
            labels = torch.arange(query_embs.size(0), device=scores.device)
            loss = loss_fct(scores / temperature, labels)
            
            # On retourne les embeddings locaux, qui ont des tailles cohérentes.
            return (loss, (query_embs, pos_doc_embs)) if return_outputs else loss
        

    # Original 
    # Compute loss either with in-batch or with only hard neg
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
    #     # feed only kwargs with 'doc_' prefix
    #     doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
    #     if "neg_doc_input_ids" in inputs:
    #         # Negative docs are not gathered across processes, so we can use them without offset
    #         neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
    #         loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
    #         return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

    #     offset = 0
    #     if self.accelerator.num_processes > 1 and self.accelerator.sync_gradients:
    #         # gather docs across all processes
    #         if num_items_in_batch is None:
    #             num_items_in_batch = inputs["doc_input_ids"].shape[0]
    #         doc_outputs = self.accelerator.pad_across_processes(doc_outputs, dim=1, pad_index=0, pad_first=True)
    #         doc_outputs = concat_all_gather(doc_outputs)
    #         rank = self.accelerator.process_index
    #         offset = rank * num_items_in_batch

    #     loss = self.loss_func(query_outputs, doc_outputs, offset=offset)

    #     return (loss, (query_outputs, doc_outputs)) if return_outputs else loss


    #Modified

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=True
    ):
        """
        Effectue une passe d'évaluation.
        Réutilise la logique de compute_loss pour garantir la cohérence.
        """
        with torch.no_grad():
            
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            predictions = outputs
            
            num_queries_in_batch = inputs["query_input_ids"].size(0)
            labels = torch.arange(num_queries_in_batch, device=loss.device)

        return loss, predictions, labels
    # Original 

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
    #     """This function is used to generate predictions and return the loss for the given inputs."""
    #     if not prediction_loss_only:
    #         raise ValueError("prediction_step is only called with prediction_loss_only=True")

    #     with torch.no_grad():
    #         # feed only kwargs with 'doc_' prefix
    #         doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
    #         query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
    #         if "neg_doc_input_ids" in inputs:
    #             neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
    #             loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
    #             return loss, None, None

    #         loss = self.loss_func(query_outputs, doc_outputs)
    #         return loss, None, None
