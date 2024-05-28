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
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'audio'"},
    )
    sentence_column_name: str = field(
        default="sentence",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'sentence'"},
    )
    min_duration_in_seconds: float = field(
        default=3584.0,
        metadata={"help": "Filter out audio files that are longer than `min_duration_in_seconds` seconds"},
    )
    max_duration_in_seconds: float = field(
        default=448512.0,
        metadata={"help": "Filter out audio files that are shorter than `max_duration_in_seconds` seconds"},
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

    vision_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )
    language_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

    ignore_ids: int = field(
        default=None,
        metadata={"help": ""},
    )
    img_token_ids: int = field(
        default=None,
        metadata={"help": ""},
    )
    num_query_seq: int = field(
        default=None,
        metadata={"help": ""},
    )
    num_query_tokens: int = field(
        default=None,
        metadata={"help": ""},
    )
    vision_projection_bias: bool = field(
        default=None,
        metadata={"help": ""},
    )

    abstract_hidden_size: int = field(
        default=None,
        metadata={"help": ""},
    )
    abstract_num_hidden_layers: int = field(
        default=None,
        metadata={"help": ""},
    )
    abstract_num_attention_heads: int = field(
        default=None,
        metadata={"help": ""},
    )
    abstract_intermediate_size: int = field(
        default=None,
        metadata={"help": ""},
    )
    abstract_attention_probs_dropout_prob: float = field(
        default=None,
        metadata={"help": ""},
    )
    abstract_layer_norm_eps: float = field(
        default=None,
        metadata={"help": ""},
    )
    abstract_encoder_hidden_size: int = field(
        default=None,
        metadata={"help": ""},
    )
