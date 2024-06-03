import gc
import importlib
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa
import torch
from data import DataCollatorForMplugOwl
from datasets import Array3D, Dataset, Features, Value, concatenate_datasets, load_dataset
from models import (
    MplugOwlAbstractorConfig,
    MplugOwlAbstractorModel,
    MplugOwlConfig,
    MplugOwlForCausalLM,
    MplugOwlProcessor,
)
from setproctitle import setproctitle
from utils import SAFE_WEIGHTS_NAME, MplugOwlPretrainingArguments

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
    HfArgumentParser,
    Trainer,
    is_torch_xla_available,
    is_wandb_available,
    set_seed,
)
from transformers import logging as hf_logging


module = importlib.import_module("transformers")
setattr(module, "MplugOwlConfig", MplugOwlConfig)
setattr(module, "MplugOwlProcessor", MplugOwlProcessor)
setattr(module, "MplugOwlForCausalLM", MplugOwlForCausalLM)
setattr(module, "MplugOwlAbstractorModel", MplugOwlAbstractorModel)
setattr(module, "MplugOwlAbstractorConfig", MplugOwlAbstractorConfig)


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


global GLOBAL_LOGGER
GLOBAL_LOGGER = None

PROMPT = """### User:
{img_token}

### Assistant:
{caption}"""


def main(train_args: MplugOwlPretrainingArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        try:
            image_ls = example["image"]
        except:
            print("error 발생! batch 스킵!")
            return {
                "pixel_values": [],
                "input_ids": [],
                train_args.length_column_name: [],
            }

        image_ls = image_ls if isinstance(image_ls, list) else [image_ls]

        if ("caption_ls" not in example) and ("question_answer" in example):
            caption_ls_ls = [[y["question"] for y in x] for x in example["question_answer"]]
        else:
            caption_ls_ls = example["caption_ls"]
        caption_ls_ls = caption_ls_ls if isinstance(caption_ls_ls, list) else [caption_ls_ls]

        query_eos_token_length = 1
        query_length = model.config.num_query_tokens + query_eos_token_length
        data = {
            "pixel_values": [],
            "input_ids": [],
            train_args.length_column_name: [],
        }
        for image, caption_ls in zip(image_ls, caption_ls_ls):
            img_outputs = processor(images=image, return_tensors="np")
            img_outputs["pixel_values"] = img_outputs["pixel_values"][0]
            for caption in random.choices(caption_ls, k=3):
                prompt = PROMPT.format(img_token="<|image|>", caption=caption)
                prompt = prompt.strip()
                prompt = f"{prompt}{processor.tokenizer.eos_token}"

                txt_outputs = processor(text=prompt, return_tensors="np")
                txt_outputs["input_ids"] = txt_outputs["input_ids"][0].tolist()

                input_ids_lenght = len(txt_outputs["input_ids"]) + query_length
                data[train_args.length_column_name].append(input_ids_lenght)
                data["pixel_values"].append(img_outputs["pixel_values"])
                data["input_ids"].append(txt_outputs["input_ids"])

        pool = pa.default_memory_pool()
        pool.release_unused()
        gc.collect()

        return data

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

    def get_mplug_owl(train_args: MplugOwlPretrainingArguments) -> Tuple[MplugOwlForCausalLM, MplugOwlProcessor]:
        if not (train_args.vision_model_name_or_path and train_args.language_model_name_or_path):
            raise ValueError

        image_processor = AutoImageProcessor.from_pretrained(train_args.vision_model_name_or_path)
        # synatra는 앞에 ['<|image|>', '▁'] 같이 들어감. 확인 필요, 보니깐 다른 special token들도 그러네.
        tokenizer = AutoTokenizer.from_pretrained(
            train_args.language_model_name_or_path,
            use_fast=False,
            padding_side="left",
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_tokens(AddedToken("<|image|>", special=True, normalized=False), special_tokens=True)

        new_vocab_size = len(tokenizer.get_vocab())

        language_config = AutoConfig.from_pretrained(
            train_args.language_model_name_or_path,
            vocab_size=new_vocab_size,
            padding_idx=tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            attn_implementation=train_args.attn_implementation,
        )
        vision_config = AutoConfig.from_pretrained(train_args.vision_model_name_or_path)

        if isinstance(vision_config, CLIPConfig):
            vision_config = vision_config.vision_config

        if isinstance(image_processor, CLIPProcessor):
            image_processor = image_processor.image_processor

            if "shortest_edge" in image_processor.size:
                image_processor.size = image_processor.size["shortest_edge"]

        vision_model = AutoModel.from_pretrained(train_args.vision_model_name_or_path, config=vision_config)
        language_model = AutoModelForCausalLM.from_pretrained(train_args.language_model_name_or_path)

        embedding = language_model.resize_token_embeddings(new_vocab_size)
        language_model.set_input_embeddings(embedding)

        if isinstance(vision_model, CLIPModel):
            vision_model = CLIPVisionModel.from_pretrained(train_args.vision_model_name_or_path, config=vision_config)

        abstractor_config = MplugOwlAbstractorConfig(
            num_hidden_layers=train_args.abstractor_num_hidden_layers,
            num_attention_heads=train_args.abstractor_num_attention_heads,
            intermediate_size=train_args.abstractor_intermediate_size,
            attention_probs_dropout_prob=train_args.abstractor_attention_probs_dropout_prob,
            layer_norm_eps=train_args.abstractor_layer_norm_eps,
            encoder_hidden_size=train_args.abstractor_encoder_hidden_size,
        )
        config = MplugOwlConfig(
            img_token_id=tokenizer.convert_tokens_to_ids("<|image|>"),
            vision_config=vision_config.to_dict(),
            language_config=language_config.to_dict(),
            abstractor_config=abstractor_config.to_dict(),
            num_query_tokens=train_args.num_query_tokens,
            vision_projection_bias=train_args.vision_projection_bias,
            ignore_id=-100,
        )

        model = MplugOwlForCausalLM(config=config)
        model.set_language_model(language_model)
        model.set_vision_model(vision_model)

        for param in model.language_model.parameters():
            param.requires_grad = False

        processor = MplugOwlProcessor(image_processor, tokenizer)

        return (model, processor)

    model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path or ""

    # load model, feature_extractor, tokenizer
    if os.path.exists(os.path.join(model_name_or_path, SAFE_WEIGHTS_NAME)):
        model = MplugOwlForCausalLM.from_pretrained(model_name_or_path)
        processor = MplugOwlProcessor.from_pretrained(model_name_or_path)
    else:
        model, processor = get_mplug_owl(train_args)

        init_save_path = os.path.join(train_args.output_dir, "init_modal")
        if os.path.exists(init_save_path):
            model.save_pretrained(init_save_path)
            processor.save_pretrained(init_save_path)

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

            features = Features(
                {
                    "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
                    "input_ids": [Value("int32")],
                    train_args.length_column_name: Value("int32"),
                }
            )
            dataset = dataset.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=column_names,
                features=features,
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

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    response_token_ids = processor.tokenizer.encode("\n\n### Assistant:\n")[5:]
    collator = DataCollatorForMplugOwl(
        processor=processor,
        img_token_ids=model.config.img_token_id,
        response_token_ids=response_token_ids,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        # eval_dataset=valid_dataset_dict,
    )
    if train_args.do_train and train_dataset:
        train(trainer)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        predict(trainer, test_dataset)


def train(trainer: Trainer) -> None:
    train_args: MplugOwlPretrainingArguments = trainer.args
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)
    # trainer 특성 때문에 save_metrics 안됨.


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


@torch.no_grad()
def predict(trainer: Trainer, test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    test_dataset_dict = dict()
    test_name_ls = test_dataset["dataset_name"]
    for dataset_name in set(test_name_ls):
        part_idx = [idx for idx, x in enumerate(test_name_ls) if x == dataset_name]
        part_dataset = test_dataset.select(part_idx, keep_in_memory=False)

        # 'jp1924/KconfSpeech-validation'
        start = dataset_name.rindex("/") + 1
        end = dataset_name.rindex("-")

        outputs = trainer.predict(part_dataset, metric_key_prefix=f"test/{dataset_name[start:]}")
        # NOTE: trainer.log를 사용하면 train/test 처럼 찍혀서 나와서 wandb로 직접 찍음
        if GLOBAL_LOGGER:
            GLOBAL_LOGGER.log(outputs.metrics)
        test_dataset_dict[dataset_name[start:end]] = part_dataset


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
