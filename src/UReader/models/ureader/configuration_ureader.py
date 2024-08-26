import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging


logger = logging.get_logger(__name__)


class UReaderAbstractorConfig(PretrainedConfig):
    model_type = "mplug_owl_visual_abstract"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        encoder_hidden_size=1024,
        shape_croping_position_type="post",
        cut_num: int = 15,
        embedding_scale: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size
        self.shape_croping_position_type = shape_croping_position_type
        self.cut_num = cut_num
        self.embedding_scale = embedding_scale

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

    def __init__(
        self,
        img_token_id: int,
        num_query_tokens: int = 64,
        ignore_id: int = -100,
        vision_projection_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        if "language_config" not in kwargs:
            raise ValueError("`language_config` can not be `None`.")

        vision_config = kwargs.pop("vision_config", {})
        language_config = kwargs.pop("language_config", {})
        abstractor_config = kwargs.pop("abstractor_config", {})

        vision_model_type = vision_config["model_type"]
        language_model_type = language_config["model_type"]

        vision_config_class = CONFIG_MAPPING[vision_model_type]
        language_config_class = CONFIG_MAPPING[language_model_type]

        self.vision_config = vision_config_class(**vision_config)
        self.language_config = language_config_class(**language_config)

        abstractor_config["encoder_hidden_size"] = self.vision_config.hidden_size
        abstractor_config["pad_token_id"] = self.pad_token_id

        self.abstractor_config = UReaderAbstractorConfig(**abstractor_config)

        self.num_query_tokens = num_query_tokens

        self.initializer_factor = 1.0
        self.initializer_range = 0.02

        self.pad_token_id = self.language_config.pad_token_id
        self.img_token_id = img_token_id
        self.ignore_id = ignore_id
        self.vision_projection_bias = vision_projection_bias

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
