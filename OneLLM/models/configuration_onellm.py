from transformers import PretrainedConfig


class OneLLMConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)