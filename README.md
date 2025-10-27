# ColPali-engine: Enhanced with Multiple Hard Negatives Support 🚀

This is a fork of the original [`illuin-tech/colpali`](https://github.com/illuin-tech/colpali) repository, enhanced to support training bi-encoder models with **multiple hard negatives** per query in a single training step.

## The Limitation in the Original Implementation 🎯

The original `ColPali-engine` is a powerful tool for training state-of-the-art bi-encoders. However, its training pipeline is designed around a triplet structure: `(query, positive_document, negative_document)`.

This design limits each training step to learning from only **one hard negative** example at a time. While effective, this can be suboptimal as the model misses the opportunity to learn from a more diverse set of negative examples simultaneously.

## Our Solution: Multiple Hard Negatives ✨

This fork overcomes that limitation by enabling the model to learn from a group of hard negatives for each query in a single forward pass. This provides a richer, more diverse training signal, which can lead to more robust and accurate retrieval models.

The changes are implemented across the three core components of the training pipeline: the **Dataset**, the **Data Collator**, and the **Trainer**.

