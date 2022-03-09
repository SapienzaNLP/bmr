import copy
import math
import random
from typing import *

import torch
from torch.nn import functional as F

from transformers.models.mbart.modeling_mbart import *


class AMRMBartForConditionalGeneration(MBartForConditionalGeneration):
    def __init__(self, config: MBartConfig, backpointer_idx=None):
        super().__init__(config)
        self._rev = None

    def init_reverse_model(self):
        rev = AMRBartForConditionalGeneration(self.model.config, self.backpointer_idx)
        rev.model.shared = self.model.shared
        rev.model.encoder = self.model.encoder
        rev.model.decoder.embed_tokens = self.model.decoder.embed_tokens
        rev.model.decoder.embed_positions = self.model.decoder.embed_positions
        self._rev = rev

    @property
    def rev(self):
        if self._rev is None:
            return self

        return self._rev

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.nll_loss(
                lm_logits.log_softmax(-1).contiguous().view(-1, lm_logits.size(-1)),
                labels.contiguous().view(-1),
                ignore_index=self.config.pad_token_id,
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
