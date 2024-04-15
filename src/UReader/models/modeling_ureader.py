from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    Blip2QFormerModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .configuration_ureader import UReaderAbstractorConfig, UReaderConfig


class UReaderPatchEmbeddings(nn.Module):

    def __init__(self, config: UReaderAbstractorConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cut_num = 15  # TODO hard code되어 있음.
        self.h_postion_patch_embedding = torch.nn.Embedding(self.cut_num, config.hidden_size)  # height
        self.w_postion_patch_embedding = torch.nn.Embedding(self.cut_num, config.hidden_size)  # width

    def forward(
        self,
        hidden_states,
        patch_positions,
    ):
        h_embedding = self.h_postion_patch_embedding(patch_positions[:, 0])
        w_embedding = self.w_postion_patch_embedding(patch_positions[:, 1])
        patch_embedding = h_embedding + w_embedding

        patch_embedding = patch_embedding[:, None, :]  # [N, D] > [N, 1, D]
        patch_embedding = patch_embedding.expand(-1, hidden_states.shape[1], -1)  # [N, 1, D] > [N, S, D]

        patch_embedding = hidden_states + patch_embedding

        return patch_embedding


# from copide Copied from transformers.models.modeling_blip_2.Blip2QFormerModel
class UReaderAbstractorModel(Blip2QFormerModel):
    def __init__(self, config: UReaderAbstractorConfig):
        super().__init__(config)

        self.patch_postion_embedding = UReaderPatchEmbeddings(config)

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
        patch_positions: Optional[torch.LongTensor] = None,  # NOTE: 이게 추가 됨
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        if (self.config.position_embedding_type == "pre") and (not self.config.vision_cut_embedding):
            patch_embedding = self.patch_postion_embedding(encoder_hidden_states, patch_positions)
            cut_index = (patch_positions == 0).all(dim=1).nonzero(as_tuple=False).squeeze(1)

            patch_groups = []
            start_index = 0
            for index in cut_index:
                patch_groups.append(patch_embedding[start_index:index].reshape(-1, patch_embedding.size(2)))
                start_index = index

            patch_groups.append(patch_embedding[start_index:].reshape(-1, patch_embedding.size(2)))

            patch_masks = [
                torch.ones(group.size(0), dtype=torch.long, device=patch_embedding.device) for group in patch_groups
            ]

            encoder_hidden_states = nn.utils.rnn.pad_sequence(patch_groups, batch_first=True, padding_value=0)
            encoder_attention_mask = nn.utils.rnn.pad_sequence(patch_masks, batch_first=True, padding_value=0)

        q_former_outputs = super().forward(
            query_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = q_former_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if (self.config.position_embedding_type == "post") and (not self.config.vision_cut_embedding):
            sequence_output = self.patch_postion_embedding(sequence_output, patch_positions)

        if not return_dict:
            return (sequence_output, pooled_output) + q_former_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=q_former_outputs.past_key_values,
            hidden_states=q_former_outputs.hidden_states,
            attentions=q_former_outputs.attentions,
            cross_attentions=q_former_outputs.cross_attentions,
        )


class UReaderForConditionalGeneration(PreTrainedModel):
    config_class = None
    supports_gradient_checkpointing = True
    base_model_prefix = "LlamaUReader"
    # 이건 확인
    _tied_weights_keys = ["lm_head.weight"]

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

        self.vision_projection = torch.nn.Linear(config.hidden_size, config.language_config.hidden_size)
        self.vision_eos_token = torch.nn.Parameter(torch.randn(1, 1, config.language_config.hidden_size))

        nn.init.trunc_normal_(self.vision_eos_token, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,
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

        if (pixel_values is not None) and (input_ids is not None):
            # NOTE: 다양한 vision encoder를 사용할 수 있기 때문에 각 값을 vision_kwargs로 받는다.
            vision_outputs = self.vision_model(pixel_values=pixel_values, **vision_kwargs)
            vision_embeds = vision_outputs.last_hidden_state

            vision_attention_mask = torch.ones(vision_embeds.size()[:-1], dtype=torch.long, device=vision_embeds.device)
            query_tokens = self.query_tokens.expand(vision_embeds.shape[0], -1, -1)

            abstractor_outputs = self.abstractor(
                query_embeds=query_tokens,
                encoder_hidden_states=vision_embeds,
                encoder_attention_mask=vision_attention_mask,
                patch_positions=patch_positions,
            )
            abstractor_last_hidden_states = abstractor_outputs[0]

            vision_token = self.vision_projection(abstractor_last_hidden_states)
            vision_eos_token = self.vision_eos_token.repeat(vision_token.shape[0], 1, 1)

            language_model_embedding_layer = self.language_model.get_input_embeddings()
            inputs_embeds = language_model_embedding_layer(input_ids)

            img_idx = 0
            for b in range(batch_size):
                start = 0
                result = []
                if len(media_token_indices[b]) > 0:
                    for i, pos in enumerate(media_token_indices[b][0]):
                        if pos > start:
                            result.append(text_embeds[b, start:pos])
                        result.append(query_features[img_idx + i])
                        start = pos + img_seq_length
                if start < text_embeds.shape[1]:
                    result.append(text_embeds[b, start:])

                img_idx += media_token_indices[b][1]
                text_chunk_embeds.append(torch.cat(result, dim=0))
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
