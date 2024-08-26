from dataclasses import dataclass, field

from setproctitle import setproctitle

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
    set_seed,
)
from transformers import logging as hf_logging


try:
    from . import MplugOwlAbstractorConfig, MplugOwlConfig, MplugOwlForCausalLM, MplugOwlProcessor
except ImportError:
    from configuration_mplug_owl import MplugOwlAbstractorConfig, MplugOwlConfig
    from modeling_mplug_owl import MplugOwlAbstractorModel, MplugOwlForCausalLM
    from processing_mplug_owl import MplugOwlProcessor


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

IMG_TOKEN = "<|image|>"


@dataclass
class ConvertArguments:
    output_dir: str = field(
        default="/root/mplug_owl_init_model",
        metadata={"help": ""},
    )

    vision_model_name_or_path: str = field(
        default="Bingsu/clip-vit-large-patch14-ko",
        metadata={"help": ""},
    )
    language_model_name_or_path: str = field(
        default="maywell/Synatra-7B-v0.3-dpo",
        metadata={"help": ""},
    )
    ignore_ids: int = field(
        default=-100,
        metadata={"help": ""},
    )
    num_query_tokens: int = field(
        default=32,
        metadata={"help": ""},
    )
    vision_projection_bias: bool = field(
        default=False,
        metadata={"help": ""},
    )
    abstractor_num_hidden_layers: int = field(
        default=6,
        metadata={"help": ""},
    )
    abstractor_num_attention_heads: int = field(
        default=16,
        metadata={"help": ""},
    )
    abstractor_intermediate_size: int = field(
        default=2048,
        metadata={"help": ""},
    )
    abstractor_attention_probs_dropout_prob: float = field(
        default=0.01,
        metadata={"help": ""},
    )
    abstractor_layer_norm_eps: float = field(
        default=1e-6,
        metadata={"help": ""},
    )
    abstractor_encoder_hidden_size: int = field(
        default=1024,
        metadata={"help": ""},
    )
    attn_implementation: str = field(
        default="flash_attn",
        metadata={"help": ""},
    )


def main(convert_args: ConvertArguments) -> None:
    if not (convert_args.vision_model_name_or_path and convert_args.language_model_name_or_path):
        raise ValueError

    image_processor = AutoImageProcessor.from_pretrained(convert_args.vision_model_name_or_path)

    # NOTE: synatra는 앞에 ['<|image|>', '▁'] 같이 들어감. 확인 필요, 보니깐 다른 special token들도 그러네.
    tokenizer = AutoTokenizer.from_pretrained(
        convert_args.language_model_name_or_path,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_tokens(AddedToken(IMG_TOKEN, special=True, normalized=False), special_tokens=True)

    new_vocab_size = len(tokenizer.get_vocab())

    language_config = AutoConfig.from_pretrained(
        convert_args.language_model_name_or_path,
        vocab_size=new_vocab_size,
        padding_idx=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        attn_implementation=convert_args.attn_implementation,
    )
    vision_config = AutoConfig.from_pretrained(convert_args.vision_model_name_or_path)

    if isinstance(vision_config, CLIPConfig):
        vision_config = vision_config.vision_config

    if isinstance(image_processor, CLIPProcessor):
        image_processor = image_processor.image_processor

        if "shortest_edge" in image_processor.size:
            image_processor.size = image_processor.size["shortest_edge"]

    vision_model = AutoModel.from_pretrained(convert_args.vision_model_name_or_path, config=vision_config)
    language_model = AutoModelForCausalLM.from_pretrained(convert_args.language_model_name_or_path)

    embedding = language_model.resize_token_embeddings(new_vocab_size)
    language_model.set_input_embeddings(embedding)

    if isinstance(vision_model, CLIPModel):
        vision_model = CLIPVisionModel.from_pretrained(convert_args.vision_model_name_or_path, config=vision_config)

    abstractor_config = MplugOwlAbstractorConfig(
        num_hidden_layers=convert_args.abstractor_num_hidden_layers,
        num_attention_heads=convert_args.abstractor_num_attention_heads,
        intermediate_size=convert_args.abstractor_intermediate_size,
        attention_probs_dropout_prob=convert_args.abstractor_attention_probs_dropout_prob,
        layer_norm_eps=convert_args.abstractor_layer_norm_eps,
        encoder_hidden_size=convert_args.abstractor_encoder_hidden_size,
    )
    config = MplugOwlConfig(
        img_token_id=tokenizer.convert_tokens_to_ids(IMG_TOKEN),
        vision_config=vision_config.to_dict(),
        language_config=language_config.to_dict(),
        abstractor_config=abstractor_config.to_dict(),
        num_query_tokens=convert_args.num_query_tokens,
        vision_projection_bias=convert_args.vision_projection_bias,
        ignore_id=-100,
    )

    model = MplugOwlForCausalLM(config=config)
    model.set_language_model(language_model)
    model.set_vision_model(vision_model)

    processor = MplugOwlProcessor(image_processor, tokenizer)

    model.save_pretrained(convert_args.output_dir)
    processor.save_pretrained(convert_args.output_dir)

    MplugOwlProcessor.from_pretrained(convert_args.output_dir)
    MplugOwlForCausalLM.from_pretrained(convert_args.output_dir)


if "__main__" in __name__:
    setproctitle("convert_owl_weights_to_hf")

    parser = HfArgumentParser([ConvertArguments])
    convert_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    set_seed(42)

    main(convert_args)
