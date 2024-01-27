from transformers import PreTrainedModel, LlamaModel, logging

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import (
    CLIPVisionModel,
    CLIPEncoder,
    CLIPVisionEmbeddings,
    CLIPAttention,
    CLIPMLP,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPooling
from .configuration_onellm import OneLLMConfig, OneLLMUniversalEncoderConfig
import torch.nn as nn
import torch
from typing import Optional, List, Union, Tuple

logger = logging.get_logger(__name__)


class CLIPAudioEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.patch_size = config.audio_patch_size
        self.stride_size = config.audio_stride_size
        # NOTE: class embedding 값은 clip의 weight를 가져다 사용해야 하는데 이걸 어떻게 구현해야 할 지 모르겠음. 고민해 볼 것
        self.patch_embedding = nn.Conv2d(
            in_channels=config.audio_num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride_size,
            bias=False,
        )

        self.num_positions = config.audio_num_positions
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False
        )


class CLIPFMRIEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        # NOTE: class embedding 값은 clip의 weight를 가져다 사용해야 하는데 이걸 어떻게 구현해야 할 지 모르겠음. 고민해 볼 것
        self.patch_embedding = nn.Linear(config.fmri_input, config.fmri_output)

        self.num_positions = config.fmri_num_positions
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype

        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.reshape(batch_size, self.embed_dim, -1)
        # [B, 1, 8196] -> [B, hidden_size, 8]

        breakpoint()  # BUG: shape애러가 발생할 수 있음.
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPIMUEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.patch_size = config.imu_num_channels
        # NOTE: class embedding 값은 clip의 weight를 가져다 사용해야 하는데 이걸 어떻게 구현해야 할 지 모르겠음. 고민해 볼 것
        self.patch_embedding = nn.Conv1d(
            in_channels=config.imu_num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            bias=False,
        )

        self.num_positions = config.imu_num_positions
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False
        )


class OneLLMPreTrainedModel(PreTrainedModel):
    config_class = OneLLMConfig
    base_model_prefix = "onellm"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, CLIPVisionEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(
                module.patch_embedding.weight, std=module.config.initializer_range * factor
            )
            nn.init.normal_(
                module.position_embedding.weight, std=module.config.initializer_range * factor
            )
        elif isinstance(module, CLIPAudioEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(
                module.patch_embedding.weight, std=module.config.initializer_range * factor
            )
            nn.init.normal_(
                module.position_embedding.weight, std=module.config.initializer_range * factor
            )
        elif isinstance(module, CLIPFMRIEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 얘만 patch_embedding로 linear를 사용함.
            nn.init.normal_(module.patch_embedding, std=module.config.initializer_range * factor)
            nn.init.normal_(
                module.position_embedding.weight, std=module.config.initializer_range * factor
            )
        elif isinstance(module, CLIPIMUEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(
                module.patch_embedding.weight, std=module.config.initializer_range * factor
            )
            nn.init.normal_(
                module.position_embedding.weight, std=module.config.initializer_range * factor
            )
        elif isinstance(module, CLIPAttention):
            in_proj_std = (
                (module.embed_dim**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPMLP):
            in_proj_std = (
                (module.config.hidden_size**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class OneLLMUniversalEncoder(nn.Module):
    def __init__(self, config: OneLLMUniversalEncoderConfig) -> None:
        super().__init__()
        self.image_embeddings = CLIPVisionEmbeddings(config)
        self.audio_embeddings = CLIPAudioEmbeddings(config)
        self.fmri_embeddings = CLIPFMRIEmbeddings(config)
        self.imu_embeddings = CLIPIMUEmbeddings(config)
        # point_embedding는 구현이 너무 어려워서 뺌, cuda하고 cpp파일 넣어야 함.
        # sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
        # class_embedding 값을 무조건 clip의 사전 학습된 weight로 바꿔놔야 함!!!!
        # sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        data_type: str = "",
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if data_type not in ["image", "audio", "imu", "fmri"]:
            raise ValueError("지원하지 않는 데이터 타입")

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if data_type == "image":
            # original clip embedding layer
            hidden_states = self.image_embeddings(pixel_values)
        elif data_type == "audio":
            hidden_states = self.audio_embeddings(pixel_values)
        elif data_type == "imu":
            hidden_states = self.imu_embeddings(pixel_values)
        elif data_type == "fmri":
            hidden_states = self.fmri_embeddings(pixel_values)

        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class OneLLMUniversalProjection(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class OneLLMForLLama(OneLLMPreTrainedModel):
    def __init__(self, config: OneLLMConfig) -> None:
        super().__init__(config)

        # config
        self.config = config
        llm_config = config.llm_config

        # clip

        # MoE

        # llm
        self.llm_embed_tokens = nn.Embedding(
            llm_config.vocab_size,
            llm_config.hidden_size,
            llm_config.padding_idx,
        )
        self.llm_layers = nn.ModuleList(
            [
                LlamaDecoderLayer(llm_config, layer_idx)
                for layer_idx in range(llm_config.num_hidden_layers)
            ]
        )
        self._use_sdpa = llm_config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = llm_config._attn_implementation == "flash_attention_2"
        self.llm_norm = LlamaRMSNorm(llm_config.hidden_size, eps=llm_config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def load_pretrained_llm(self, model_name_or_path: str, **kwargs):
        # 모델을 처음 학습 시키는 경우
        # 걱정되는 부분, deepspeed3로 학습 시킬 때 __init__부분은 deepspeed로 초기화 되지 않았을 거지만
        # from_pretrained로 불러들이는 모델은 deepspeed로 초기화가 될 것이기 때문에 이게 학습에 어떤 영향을 미칠지 알 수 없음.
        llm = LlamaModel.from_pretrained(model_name_or_path, **kwargs)

        self.llm_embed_tokens = llm.embed_tokens
        self.llm_layers = llm.layers

        self._use_sdpa = llm._use_sdpa
        self._use_flash_attention_2 = llm._use_flash_attention_2
        self.llm_norm = llm.norm

    def load_pretrained_clip(self, model_name_or_path: str, **kwargs):
        # 모델을 처음 학습 시키는 경우
        # 그리고 불러온 모델을 다시 set_attr하는 거다 보니 메모리 파편화 및 여러 문제가 발생할 여지가 다분히 존재함.
        clip = CLIPVisionModel.from_pretrained(model_name_or_path, **kwargs)

        self.clip_embeddings = clip.vision_model.clip_embeddings
        self.clip_pre_layrnorm = clip.vision_model.clip_pre_layrnorm
        self.clip_encoder = clip.vision_model.clip_encoder
        self.clip_post_layernorm = clip.vision_model.clip_post_layernorm

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        # 이게 clip하고 어떻게 될지 모르겠음. 일단 넣어봄
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            )
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if pixel_values is not None:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        return
