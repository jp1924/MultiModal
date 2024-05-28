import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from configuration_ureader import UReaderAbstractorConfig, UReaderConfig

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    CLIPVisionConfig,
    CLIPVisionModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->UReader
class UReaderCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class UReaderPatchEmbeddings(nn.Module):
    def __init__(self, config: UReaderAbstractorConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cut_num = 15  # TODO hard code되어 있음.
        self.h_postion_patch_embedding = torch.nn.Embedding(self.cut_num, config.hidden_size)  # height
        self.w_postion_patch_embedding = torch.nn.Embedding(self.cut_num, config.hidden_size)  # width

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        patch_positions: torch.FloatTensor,
    ) -> torch.FloatTensor:
        h_embedding = self.h_postion_patch_embedding(patch_positions[:, 0])
        w_embedding = self.w_postion_patch_embedding(patch_positions[:, 1])
        patch_embedding = (h_embedding + w_embedding) * 0.5

        patch_embedding = patch_embedding[:, None, :]  # [N, D] > [N, 1, D]
        patch_embedding = patch_embedding.expand(-1, hidden_states.shape[1], -1)  # [N, 1, D] > [N, S, D]

        patch_embedding = hidden_states + patch_embedding

        return patch_embedding


class UReaderAbstractorMLP(nn.Module):
    def __init__(self, config: UReaderAbstractorConfig):
        super().__init__()
        self.config = config
        in_features = config.hidden_size
        self.act = nn.SiLU()

        self.w1 = nn.Linear(in_features, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, in_features)
        self.w3 = nn.Linear(in_features, config.intermediate_size)
        self.ffn_ln = LayerNormFp32(config.intermediate_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class UReaderAbstractorMultiHeadAttention(nn.Module):
    def __init__(self, config: UReaderAbstractorConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class UReaderAbstractorCrossOutput(nn.Module):
    def __init__(self, config: UReaderAbstractorConfig):
        super().__init__()
        dim = config.hidden_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = LayerNormFp32(dim)
        self.mlp = UReaderAbstractorMLP(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor


class UReaderAbstractorAttention(nn.Module):
    def __init__(self, config: UReaderAbstractorConfig):
        super().__init__()
        self.attention = UReaderAbstractorMultiHeadAttention(config)
        self.output = UReaderAbstractorCrossOutput(config)

        self.norm1 = LayerNormFp32(config.hidden_size)
        self.normk = LayerNormFp32(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # HACK we apply norm on q and k
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=-1)
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class UReaderAbstractorLayer(nn.Module):
    def __init__(self, config: UReaderAbstractorConfig, layer_idx: int):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.layer_idx = layer_idx

        self.crossattention = UReaderAbstractorAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be given for cross-attention layers")
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=output_attentions,
        )
        query_attention_output = cross_attention_outputs[0]

        outputs = (query_attention_output,)
        return outputs


class UReaderAbstractorEncoder(nn.Module):
    def __init__(self, config: UReaderAbstractorConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [UReaderAbstractorLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        use_cache: Optional[bool] = None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class UReaderPreTrainedModel(PreTrainedModel):
    config_class = UReaderConfig
    base_model_prefix = "ureader"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "UReaderAbstractorModel",
        "Parameter",
    ]
    _skip_keys_device_placement = "past_key_values"
    _tied_weights_keys = []

    def _init_weights(self, module) -> None:
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa

    @property
    def _supports_flash_attn_2(self) -> bool:
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_flash_attn_2


class UReaderAbstractorModel(UReaderPreTrainedModel):
    def __init__(self, config: UReaderAbstractorConfig) -> None:
        super().__init__(config)
        self.config = config

        self.encoder = UReaderAbstractorEncoder(config)
        self.patch_postion_embedding = UReaderPatchEmbeddings(config)

        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask.to(device)

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,  # NOTE: 이게 추가 됨
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = query_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        device = query_embeds.device

        if self.config.shape_croping_position_type == "pre":
            encoder_hidden_states = self.patch_postion_embedding(encoder_hidden_states, patch_positions)

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.encoder(
            query_embeds,
            head_mask=head_mask,
            use_cache=use_cache,
            return_dict=return_dict,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if self.config.shape_croping_position_type == "post":
            sequence_output = self.patch_postion_embedding(sequence_output, patch_positions)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class UreaderForCausalLM(UReaderPreTrainedModel):
    def __init__(
        self,
        config: UReaderConfig,
        vision_model: Optional[PreTrainedModel] = None,
        language_model: Optional[PreTrainedModel] = None,
    ) -> None:
        super().__init__(config)
        self.config = config

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)
        if language_model is None:
            language_model = AutoModelForCausalLM.from_config(config.language_config)

        # Copied from BLIP-2
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys.extend([f"language_model.{k}" for k in language_model._tied_weights_keys])

        # 이건 내가 만든거, TODO: 근데 vision에 tied_weights가 필요함?
        if vision_model._tied_weights_keys is not None:
            self._tied_weights_keys.extend([f"vision_model.{k}" for k in vision_model._tied_weights_keys])

        self._no_split_modules.extend(vision_model._no_split_modules)
        self._no_split_modules.extend(language_model._no_split_modules)

        self.vision_model = vision_model
        self.language_model = language_model

        # Copied from BLIP-2
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.abstractor_config.hidden_size))
        self.abstractor = UReaderAbstractorModel(config.abstractor_config)

        self.vision_projection = nn.Linear(
            config.vision_config.hidden_size,
            config.language_config.hidden_size,
            bias=config.vision_projection_bias,
        )

        self.vision_eos_token = nn.Parameter(torch.zeros(1, 1, config.language_config.hidden_size))
        nn.init.trunc_normal_(self.vision_eos_token, mean=0.0, std=self.config.initializer_range)

        # 여기선 model weight를 불러오는게 아니라 단순 뼈대만 불러옴. weight도 같이 불러올러면 from_pretrained를 하거나 load_state_dict을 해야 함.
        self.post_init()
