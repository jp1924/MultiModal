import copy
import os
from typing import Union

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.chinese_clip.configuration_chinese_clip import (
    ChineseCLIPVisionConfig,
)
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

MPLUG_OWL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "MAGAer13/mplug-owl-llama-7b": "https://huggingface.co/MAGAer13/mplug-owl-llama-7b/resolve/main/config.json",
    # See all MplugOwl models at https://huggingface.co/models?filter=mplug_owl
}

VISION_MODEL_CONFIGS = {
    "clip_vision_model": CLIPVisionConfig,
    "chinese_clip_vision_model": ChineseCLIPVisionConfig,
    "siglip_vision_model": SiglipVisionConfig,
}


class UReaderAbstractorConfig(PretrainedConfig):
    model_type = "ureader_abstractor"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        cross_attention_frequency=2,
        encoder_hidden_size=1408,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the visual_abstractor config dict if we are loading from MplugOwlConfig
        if config_dict.get("model_type") == "mplug-owl":
            config_dict = config_dict["abstractor_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class UReaderConfig(PretrainedConfig):
    model_type = "ureader"
    is_composition = True

    def __init__(self, num_query_tokens=64, **kwargs):
        super().__init__(**kwargs)

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        if "language_config" not in kwargs:
            raise ValueError("`language_config` can not be `None`.")

        vision_config = kwargs.pop("vision_config")
        language_config = kwargs.pop("language_config")
        abstractor_config = kwargs.pop("abstractor_config", {})

        vision_model_type = vision_config.pop("model_type")
        language_model_type = language_config.pop("model_type")

        vision_config_class = VISION_MODEL_CONFIGS.get(vision_model_type)
        if vision_config_class is not None:
            self.vision_config = vision_config_class(**vision_config)
        else:
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config)
            if hasattr(self.vision_config, "vision_config"):
                self.vision_config = self.vision_config.vision_config

        self.language_config = AutoConfig.for_model(language_model_type, **language_config)
        self.abstractor_config = UReaderAbstractorConfig(**abstractor_config)

        self.num_query_tokens = num_query_tokens
        self.abstractor_config.encoder_hidden_size = self.vision_config.hidden_size

        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_text_configs(
        cls,
        vision_config: PretrainedConfig,
        language_config: PretrainedConfig,
        abstractor_config: UReaderAbstractorConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            language_config=language_config.to_dict(),
            abstractor_config=abstractor_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["language_config"] = self.language_config.to_dict()
        output["abstractor_config"] = self.abstractor_config.to_dict()
        output["model_type"] = self.__class__.model_type

        return output
