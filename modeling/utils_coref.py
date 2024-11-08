# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import glob


logger = logging.getLogger(__name__)
logging.getLogger('filelock').setLevel(logging.ERROR)

@dataclass
class CorefInputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
        reference: (Optional) list. List specifying for each word the index of the word
        it refers to, if there is one. #marco
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]
    refs: Optional[List[int]]
    book_start: bool


@dataclass
class CorefInputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    ref_ids: Optional[List[int]] = None
    book_start: Optional[List[bool]] = None


class Split(Enum):
    train = "train"
    inference = "inference"
    dev = "dev"
    test = "test"


class TokenClassificationCorefTask:
    def read_examples_from_folder(self, tokenizer: PreTrainedTokenizer, data_dir, mode: Union[Split, str]) -> List[CorefInputExample]:
        raise NotImplementedError

    def get_labels(self, path: str) -> List[str]:
        raise NotImplementedError

    def convert_examples_to_features(
        self,
        examples: List[CorefInputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        pad_token_ref_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ) -> List[CorefInputFeatures]:
        """Loads a data file into a list of `CorefInputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # TODO clean up all this to leverage built-in features of tokenizers

        label_map = {label: i for i, label in enumerate(label_list)}
        ref_keywords = [pad_token_ref_id, max_seq_length-1]

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10_000 == 0:
                logger.debug("Writing example %d of %d", ex_index, len(examples))

            tokens = []
            label_ids = []
            ref_ids = []
            ref_ids_words = []
            w_idx_of_token = []
            assert(len(example.labels) == len(example.refs))
            for i, (word, label, ref) in enumerate(zip(example.words, example.labels, example.refs)):
                word_tokens = tokenizer.tokenize(word)
                ref_id = int(ref)-1 if ref not in ref_keywords else int(ref)
                # ref_tokens = tokenizer.tokenize(ref)
                # if len(ref_tokens)<4:
                #     ref_tokens+= ['▁-']*(4-len(ref_tokens))
                # else:
                #     ref_tokens = ref_tokens[:4]
                # assert(len(ref_tokens)==4)
                # exit()
                w_idx_of_token.extend([i]*max(len(word_tokens), 1))
                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                    # ref_ids.extend(ref_tokens + [tokenizer.decode(pad_token)] * 4 * (len(word_tokens) - 1))
                    ref_ids_words.extend([ref_id] + [pad_token_ref_id] * (len(word_tokens) - 1))


            #transform reference indices from word indices to token indices
            for i in range(len(ref_ids_words)):
                if ref_ids_words[i] in ref_keywords:
                    ref_ids.append(ref_ids_words[i])
                else:
                    ref_ids.append(w_idx_of_token.index(ref_ids_words[i]))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                ref_ids = ref_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            ref_ids += [pad_token_ref_id]
            # ref_ids += [tokenizer.decode(pad_token)]*4
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
                ref_ids += [pad_token_ref_id]
                # ref_ids += [tokenizer.decode(pad_token)]*4
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                # ref_ids += [tokenizer.decode(pad_token)]*4
                ref_ids += [pad_token_ref_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                # ref_ids = [tokenizer.decode(pad_token)]*4 + ref_ids
                ref_ids = [pad_token_ref_id] + ref_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # ref_ids = tokenizer.convert_tokens_to_ids(ref_ids)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # input_ref_mask = [1 if mask_padding_with_zero else 0] * len(ref_ids)
            # outputput_ref_mask = [1 if mask_padding_with_zero else 0] * len(ref_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                # input_ref_mask = ([0 if mask_padding_with_zero else 1] * 4*padding_length) + input_ref_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                # ref_ids = ([[pad_token_ref_id]] * padding_length) + ref_ids
                ref_ids = ([pad_token_ref_id] *padding_length) + ref_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                # input_ref_mask += [0 if mask_padding_with_zero else 1] * 4*padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                ref_ids += [pad_token_ref_id] * padding_length

            # stretched_labels = [x for x in label_ids for _ in range(4)]
            # output_ref_mask = [i==-100 for i in stretched_labels]
            book_start = [example.book_start]*max_seq_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            # assert len(input_ref_mask) == max_seq_length*4
            # assert len(output_ref_mask) == max_seq_length*4
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(ref_ids) == max_seq_length
            assert len(book_start) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
                logger.info("ref_ids: %s", " ".join([str(x) for x in ref_ids]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                CorefInputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids, ref_ids=ref_ids, book_start=book_start
                )
            )
        return features

class COREF(TokenClassificationCorefTask):
    def __init__(self, titles, no_ref_idx=0, label_idx=1, coref_idx=-1, to_replace_with_O=[]):
        # in NER datasets, the last column is usually reserved for NER label
        # referent index is the second to last column (#marco)
        self.label_idx = label_idx
        self.coref_idx = coref_idx
        self.no_ref_idx = no_ref_idx
        self.titles = titles
        self.to_replace_with_O = to_replace_with_O

    def read_examples_from_folder(self, tokenizer:AutoTokenizer,  data_dir, mode: Union[Split, str]) -> List[CorefInputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        guid_index = 1
        examples = []
        for file_path in sorted(glob.glob(os.path.join('.', data_dir+'/*.tsv'))):
            if mode!="inference" and not any(x in file_path for x in self.titles[mode]) :
                #les titres correspondant ÃƒÂ  ce split ne comportent pas ce titre
                #autrement dit, titre non intÃƒÂ©ressant pour ce split
                #on saute
                continue
            with open(file_path, encoding="utf-8") as f:
                words = []
                labels = []
                refs = []
                book_start = True
                for l in f:
                    line = l
                    for label in self.to_replace_with_O:
                        line = line.replace(label,'O')
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if words:
                            examples.append(CorefInputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, refs=refs, book_start=book_start))
                            book_start=False
                            guid_index += 1
                            words = []
                            labels = []
                            refs = []
                    else:
                        splits = [elem for elem in line[:-1].split("\t") if elem!='O']
                        words.append(splits[0])
                        if len(splits) > 1:
                            labels.append(splits[self.label_idx] if len(splits)>2 else 'O')
                            ref = splits[self.coref_idx] if len(splits)>2 else '-'
                            refs.append(self.no_ref_idx if ref=='-' else ref)
                        else:
                            labels.append("O")
                            refs.append(self.no_ref_idx)
                if words:
                    examples.append(CorefInputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, refs=refs, book_start=book_start))
                    book_start=False
        return examples



class TokenClassificationCorefDataset(Dataset):
    features: List[CorefInputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        token_classification_task: TokenClassificationCorefTask,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        if isinstance(mode, Split):
            mode = mode.value
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length)),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.debug(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.debug(f"Creating features from dataset file at {data_dir}")
                examples = token_classification_task.read_examples_from_folder(tokenizer, data_dir, mode)
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = token_classification_task.convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    cls_token_at_end=bool(model_type in ["xlnet"]),
                    # xlnet has a cls token at the end
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> CorefInputFeatures:
        return self.features[i]
