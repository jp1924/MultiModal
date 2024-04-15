from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaModel,
    PreTrainedModel,
)

from .configuration_ureader import UReaderAbstractorConfig, UReaderConfig


class UReaderAbstractorEmbeddings(nn.Module):

    def __init__(self, config: UReaderAbstractorConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cut_num = 15  # TODO hard code되어 있음.
        self.height_embedding = torch.nn.Embedding(self.cut_num, config.hidden_size)
        self.width_embedding = torch.nn.Embedding(self.cut_num, config.hidden_size)

    def forward(
        self,
        query_embeds,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        patch_positions=None,
        return_dict=None,
    ):
        if self.patch_pos_embed_type == "pre":
            # 根据patch_positions把Patch重新组织 打padding
            if self.enable_vit_cut_embedding:
                # 这里就不加patch_embedding了
                patch_embedding = encoder_hidden_states
            else:
                patch_embedding = (
                    self.cut_patch_embedding_h(patch_positions[:, 0])
                    + self.cut_patch_embedding_w(patch_positions[:, 1])
                ) * 0.5
                patch_embedding = einops.repeat(
                    patch_embedding, "N D -> N num_token D", num_token=encoder_hidden_states.shape[1]
                )
                patch_embedding = encoder_hidden_states + patch_embedding
            cut_index = (patch_positions == 0).all(dim=1).nonzero().squeeze(1).tolist()
            patch_group = [
                patch_embedding[cut:] if ci == len(cut_index) - 1 else patch_embedding[cut : cut_index[ci + 1]]
                for ci, cut in enumerate(cut_index)
            ]
            patch_group = [einops.rearrange(_, "num_patch num_token D -> (num_patch num_token) D") for _ in patch_group]
            patch_mask = [torch.ones(_.shape[0], dtype=torch.long, device=patch_embedding.device) for _ in patch_group]
            encoder_attention_mask = pad_sequence(patch_mask, batch_first=True, padding_value=0)
            encoder_hidden_states = pad_sequence(
                patch_group, batch_first=True, padding_value=0.0
            )  # -> B (num_patch num_token + pad) D

        return


class UReaderPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UReaderConfig
    base_model_prefix = "UReader"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
        r"language_model.lm_head.weight",
    ]
    _no_split_modules = [
        "MplugOwlVisionEncoderLayer",
        "LlamaDecoderLayer",
        "MplugOwlVisualAbstractorLayer",
        "LlamaForCausalLM",
        "Parameter",
    ]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, MplugOwlVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=factor)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            raise ValueError
            nn.init.trunc_normal_(module.data, mean=0.0, std=factor)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MplugOwlVisionEncoder):
            module.gradient_checkpointing = value


# from copide Copied from transformers.models.modeling_blip_2.Blip2QFormerModel
class UReaderAbstractorModel(UReaderPreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: UReaderAbstractorConfig):
        super().__init__(config)
        self.config = config

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = Blip2QFormerEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device (`torch.device`):
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
        return extended_attention_mask

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.layernorm(query_embeds)
        embedding_output = self.dropout(embedding_output)

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

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


class UReaderModel(PreTrainedModel):
    _tied_weights_keys = []

    def __init__(
        self,
        config: UReaderConfig,
        vision_model: Optional[PreTrainedModel] = None,
        language_model: Optional[PreTrainedModel] = None,
    ) -> None:
        self.config = config

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)

        if language_model is None:
            language_model = AutoModel.from_config(config.language_config)

        # Copied from BLIP-2
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys.extend([f"language_model.{k}" for k in language_model._tied_weights_keys])

        # 이건 내가 만든거
        if vision_model._tied_weights_keys is not None:
            self._tied_weights_keys.extend([f"vision_model.{k}" for k in vision_model._tied_weights_keys])

        # Copied from BLIP-2
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.abstractor_config.hidden_size))

        self.abstractor = UReaderAbstractorModel(config.abstractor_config)
        self.vision_model = vision_model
        self.language_model = language_model

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_kwargs: Optional[Dict[str, Any]] = {},
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is not None:
            # NOTE: 다양한 vision encoder를 사용할 수 있기 때문에 각 값을 vision_kwargs로 받는다.
            vision_outputs = self.vision_model(pixel_values=pixel_values, **vision_kwargs)
            vision_embeds = vision_outputs.last_hidden_state

            vision_attention_mask = torch.ones(vision_embeds.size()[:-1], dtype=torch.long, device=vision_embeds.device)
            query_tokens = self.query_tokens.expand(vision_embeds.shape[0], -1, -1)

            query_features = self.abstractor(
                query_embeds=query_tokens,
                encoder_hidden_states=vision_embeds,
                encoder_attention_mask=vision_attention_mask,
                patch_positions=patch_positions,
            )

        # NOTE: inputs_embeds가 input_ids보다 우선순위가 높아서 inputs_embeds`만` 사용함.
        language_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class LlamaUReaderForCausalLM(PreTrainedModel):
    config_class = None
    supports_gradient_checkpointing = True
    base_model_prefix = "LlamaUReader"
    # 이건 확인
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        vision_model: Optional[PreTrainedModel],
        language_model: Optional[PreTrainedModel],
    ) -> None:

        self.model = UReaderModel(config, vision_model, language_model)

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        vision_model: Optional[PreTrainedModel],
        language_model: Optional[PreTrainedModel],
    ) -> None:

        self.model = UReaderModel(config, vision_model, language_model)
