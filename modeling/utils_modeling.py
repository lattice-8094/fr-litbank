import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from numpy import sqrt
import random

from transformers import RobertaModel, RobertaPreTrainedModel, CamembertConfig, XLMRobertaConfig, FlaubertConfig,FlaubertModel
from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple



@dataclass
class TokenClassifierCorefOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    ner_loss: Optional[torch.FloatTensor] = None
    coref_loss: Optional[torch.FloatTensor] = None
    ner_logits: torch.FloatTensor = None
    coref_logits: torch.FloatTensor = None
    key_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForCoreference(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.no_ref_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        
        #self.query_encoder = nn.Linear(config.hidden_size, config.hidden_size) 
        self.key_encoder = RobertaLayer(config) 
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        coref_attention_mask=None,
        coref_output_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ref_ids=None,
        book_start=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bs, seq_len = input_ids.size()
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        output = self.dropout(outputs['last_hidden_state'])
        ner_output = output
        ner_logits = self.classifier(ner_output)
        query_output = output
        #query_output = self.query_encoder(query_output)
        key_output = output
        key_output = self.key_encoder(key_output)[0]

        no_ref_key = self.no_ref_classifier(key_output[:,0,:]).unsqueeze(1)
        no_ref_key = self.dropout(no_ref_key)
        k = torch.cat([key_output[:,1:,:], no_ref_key], axis=1)

        res = torch.matmul(query_output, k.transpose(1,2))
        mask = torch.tril(torch.ones_like(res), diagonal=1)==0
        mask = torch.cat([mask[:,:,1:], mask[:,:,:1]], axis=2)
        coref_logits = res.masked_fill(mask, float('-inf'))

        ner_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                ner_loss = loss_fct(active_logits, active_labels)


                active_loss_coref = attention_mask.view(-1) == 1
                active_coref_logits = coref_logits.reshape(-1, seq_len)
                active_ref_ids = torch.where(
                    active_loss_coref, ref_ids.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ref_ids)
                )
                coref_loss = loss_fct(active_coref_logits, active_ref_ids)
            else:
                ner_loss = loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))
                coref_loss = loss_fct(coref_logits.view(-1, seq_len), ref_ids.view(-1))

        if not return_dict:
            output = (ner_logits+coref_loss,) + outputs[2:]
            return ((ner_loss+coref_loss,) + output) if ner_loss is not None else output

        return TokenClassifierCorefOutput(
            ner_loss=ner_loss,
            coref_loss=coref_loss,
            ner_logits=ner_logits,
            key_output=key_output,
            coref_logits = coref_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CamembertForCoreference(RobertaForCoreference):
    config_class = CamembertConfig

#class XLMRobertaForCoreference(RobertaForCoreference):
#    config_class = XLMRobertaConfig
#
#class FlaubertForCoreference(RobertaForCoreference):
#    config_class = FlaubertConfig

#class FlaubertForCoreference(XLMRobertaForCoreference):
#    config_class = FlaubertConfig
#    def __init__(self, config):
#        super().__init__(config)
#        self.transformer = FlaubertModel(config)
#        # Initialize weights and apply final processing
#        self.post_init()
