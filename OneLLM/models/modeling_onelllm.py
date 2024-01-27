from transformers import PreTrainedModel, LlamaModel, logging

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.clip.modeling_clip import (
    CLIPVisionModel,
    CLIPEncoder,
    CLIPVisionEmbeddings,
    CLIPAttention,
    CLIPMLP,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from .configuration_onellm import OneLLMConfig
import torch.nn as nn
import torch
from typing import Optional, List

logger = logging.get_logger(__name__)


class CLIPAudioEmbeddings(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class CLIPFMRIEmbeddings(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class CLIPPointEmbeddings(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


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
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(
                module.patch_embedding.weight, std=module.config.initializer_range * factor
            )
            nn.init.normal_(
                module.position_embedding.weight, std=module.config.initializer_range * factor
            )
        elif isinstance(module, CLIPAttention):
            factor = self.config.initializer_factor
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
            factor = self.config.initializer_factor
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


class OneLLMUniversalProjection(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class OneLLMForLLama(OneLLMPreTrainedModel):
    def __init__(self, config: OneLLMConfig) -> None:
        super().__init__(config)

        # config
        self.config = config
        clip_config = config.clip
        llm_config = config.llm_config

        # clip
        self.image_embeddings = CLIPVisionEmbeddings(clip_config)
        self.clip_pre_layrnorm = nn.LayerNorm(
            clip_config.hidden_size,
            eps=clip_config.layer_norm_eps,
        )
        self.clip_encoder = CLIPEncoder(clip_config)
        self.clip_post_layernorm = nn.LayerNorm(
            clip_config.hidden_size,
            eps=clip_config.layer_norm_eps,
        )

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
