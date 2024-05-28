import os
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, concatenate_datasets, load_dataset
from models import (
    MplugOwlAbstractorConfig,
    MplugOwlConfig,
    MplugOwlForCausalLM,
    MplugOwlProcessor,
)
from setproctitle import setproctitle
from utils import MplugOwlPretrainingArguments

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    CLIPVisionModel,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_xla_available,
    is_wandb_available,
    set_seed,
)
from transformers import logging as hf_logging


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

global GLOBAL_LOGGER
GLOBAL_LOGGER = None


def main(train_args: MplugOwlPretrainingArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        return {
            "labels": normalized_sentence_ls,
            "input_values": normalized_audio_ls,
            "length": length_ls,
        }

    def collect_dataset(prefix_ls: List[str]) -> Optional[Dataset]:
        if not prefix_ls:
            return None

        data_ls = list()
        for prefix in prefix_ls:
            check_key: str = lambda key: (prefix in key)
            filter_data = [
                concatenate_datasets(data_dict.pop(key)) for key in list(data_dict.keys()) if check_key(key)
            ]
            data_ls.extend(filter_data)
        dataset = concatenate_datasets(data_ls)
        dataset.set_format("torch")

        return dataset

    def set_wandb() -> None:
        # TODO: 이건 나중에 args로 바꿀 것
        GLOBAL_LOGGER.run.log_code(
            "/root/workspace",
            include_fn=lambda path: path.endswith(".py") or path.endswith(".json"),
        )
        # logging args
        combined_dict = {**train_args.to_dict()}
        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}

        GLOBAL_LOGGER.config.update(combined_dict, allow_val_change=True)

        # set default metrics
        if getattr(GLOBAL_LOGGER, "define_metric", None):
            GLOBAL_LOGGER.define_metric("train/global_step")
            GLOBAL_LOGGER.define_metric("*", step_metric="train/global_step", step_sync=True)

        # set model watch
        _watch_model = os.getenv("WANDB_WATCH", "false")
        if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
            GLOBAL_LOGGER.watch(model, log=_watch_model, log_freq=max(100, train_args.logging_steps))
        GLOBAL_LOGGER.run._label(code="transformers_trainer")

    model_path = train_args.resume_from_checkpoint or train_args.model_name_or_path

    # load model, feature_extractor, tokenizer
    if os.path.exists(os.path.join(model_path, SAFE_WEIGHTS_NAME)):
        model = MplugOwlForCausalLM.from_pretrained(model_path)
        processor = MplugOwlProcessor.from_pretrained(model_path)
    else:
        vision_model = AutoModel.from_pretrained(train_args.vision_model_name_or_path)
        language_model = AutoModelForCausalLM.from_pretrained(train_args.language_model_name_or_path)
        if isinstance(vision_model, CLIPModel):
            vision_model = CLIPVisionModel.from_pretrained(train_args.vision_model_name_or_path)

        abstractor_config = MplugOwlAbstractorConfig()
        config = MplugOwlConfig(
            attn_implementation=train_args.attn_implementation,
            vision_config=vision_model.config,
            language_config=language_model.config,
            abstractor_config=abstractor_config,
        )

        model = MplugOwlForCausalLM(
            config=config,
            vision_model=vision_model,
            language_model=language_model,
        )

        model.language_model
        model.vision_model

        tokenizer = AutoTokenizer.from_pretrained(train_args.language_model_name_or_path)
        image_processor = AutoImageProcessor.from_pretrained(train_args.vision_model_name_or_path)
        processor = MplugOwlProcessor(image_processor, tokenizer)

    # NOTE: Trainer에서 자동으로 해줌, 하지만 확인을 위해 이렇게 선언 함.
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # set logger
    if GLOBAL_LOGGER and (train_args.local_rank == 0):
        set_wandb()

    # load dataset & preprocess
    data_dict = dict()
    for dataset_name in train_args.dataset_names:
        logger.info(f"load-{dataset_name}")
        dataset = load_dataset(dataset_name)

        # DatasetDict이라서 이런식으로 해줘야 함.
        column_names = set(sum(dataset.column_names.values(), []))
        with train_args.main_process_first(desc="data preprocess"):
            cache_file_name = None
            if train_args.cache_file_name:
                get_cache_path: str = lambda x: os.path.join(
                    train_args.cache_dir,
                    f"{name}-{x}_{train_args.cache_file_name}",
                )
                name = dataset_name.split("/")[-1]
                cache_file_name = {x: get_cache_path(x) for x in dataset}

            # NOTE: finetune에서 사용할 데이터 Pretrain에서 전처리 함
            # 만약 순수 음성만 넣을 거라면 sentence 부분을 ""로 비워든 상태로 돌리면 정상적으로 진행 됨
            dataset = dataset.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=column_names,
                desc=f"preprocess-{dataset_name}",
            )
        for data_key in dataset:
            if data_key not in data_dict:
                data_dict[data_key] = []

            specific_dataset = dataset[data_key]

            added_data = [f"{dataset_name}-{data_key}"] * len(specific_dataset)
            specific_dataset = specific_dataset.add_column("dataset_name", added_data)

            data_dict[data_key].append(specific_dataset)

    train_dataset = None
    if train_args.do_train:
        train_dataset = collect_dataset(train_args.train_dataset_prefix)
        if (train_args.local_rank == 0) and train_dataset:
            logger.info("train_dataset")
            logger.info(train_dataset)

    valid_dataset = None
    if train_args.do_eval:
        valid_dataset = collect_dataset(train_args.valid_dataset_prefix)
        if (train_args.local_rank == 0) and valid_dataset:
            logger.info("valid_dataset")
            logger.info(valid_dataset)

    test_dataset = None
    if train_args.do_predict:
        test_dataset = collect_dataset(train_args.test_dataset_prefix)
        if (train_args.local_rank == 0) and test_dataset:
            logger.info("test_dataset")
            logger.info(test_dataset)

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset_dict,
    )


if "__main__" in __name__:
    parser = HfArgumentParser([MplugOwlPretrainingArguments])
    train_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    check_wandb = ("wandb" in train_args.report_to) and (train_args.local_rank == 0)
    if is_wandb_available() and check_wandb:
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_RUN_GROUP"),
            name=train_args.run_name,
            save_code=True,
        )
        GLOBAL_LOGGER = wandb

    main(train_args)

    if GLOBAL_LOGGER:
        GLOBAL_LOGGER.finish()
