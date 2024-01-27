from transformers import PretrainedConfig, CLIPConfig


class OneLLMUniversalEncoderConfig(PretrainedConfig):
    # TODO: 여기에 PretrainedConfig를 사용할 지 말지는 고민 중, 나중에 돌려보고 뺄지 말지 결정할 것
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OneLLMUniversalProjectionConfig(PretrainedConfig):
    # TODO: 여기에 PretrainedConfig를 사용할 지 말지는 고민 중, 나중에 돌려보고 뺄지 말지 결정할 것ㅁ
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        projection_dim: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 32,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_factor: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.audio_num_channels = 1
        self.audio_num_positions = 1213
        self.audio_patch_size = 16
        self.audio_stride_size = 10

        self.imu_num_channels = 6
        self.imu_num_positions = 392
        self.imu_patch_size = 10

        self.fmri_input = 15724
        self.fmri_output = 8192
        self.fmri_num_positions = 9


class OneLLMConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
