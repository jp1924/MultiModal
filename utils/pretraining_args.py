from dataclasses import dataclass, field
from typing import List

from transformers import TrainingArguments


@dataclass
class MplugOwlPretrainingArguments(TrainingArguments):
    # data
    dataset_names: List[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batched: bool = field(
        default=True,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    train_dataset_prefix: List[str] = field(
        default="train",
        metadata={"help": ""},
    )
    valid_dataset_prefix: List[str] = field(
        default="validation",
        metadata={"help": ""},
    )
    test_dataset_prefix: List[str] = field(
        default="eval_other",
        metadata={"help": ""},
    )
    valid_exclude_ls: List[str] = field(
        default=None,
        metadata={"help": ""},
    )
    valid_truncate_num: int = field(
        default=3000,
        metadata={"help": ""},
    )
    split_valid: bool = field(
        default=False,
        metadata={"help": ""},
    )
    cache_file_name: str = field(
        default=None,
        metadata={"help": "Path to cached file name"},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    # model
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )
