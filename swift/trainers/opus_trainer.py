from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_utils import has_length

from swift.trainers import Seq2SeqTrainer
from swift.dataset import IndexDatasetWrapper
from swift.dataset import OpusSampler


class OpusTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_dataset = IndexDatasetWrapper(self.train_dataset)
        self.sample_indices = []
        self.sampler = None


    def _prepare_inputs(self, inputs):
        self.sample_indices.extend(inputs.get('sample_idx'))
        return super()._prepare_inputs(inputs)


    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        sampler = OpusSampler(train_dataset)
        self.sampler = sampler

        return sampler


    def get_train_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )
