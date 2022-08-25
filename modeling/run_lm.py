import argparse
import glob
import logging
import math
import os
import sys
import torch
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple, NewType, TextIO, Union, Any
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter
from utils_chunking import chunk_brat, write_chunk_and_all_predictions


import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch import nn

from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from utils_coref import Split, TokenClassificationCorefDataset, TokenClassificationCorefTask, CorefInputExample
from utils_ner import TokenClassificationDataset, NER, InputExample
from utils_modeling import CamembertForCoreference
from transformers.tokenization_utils_base import BatchEncoding

import datasets
from datasets import ClassLabel, load_dataset
from utils_eval import run_evaluation

InputDataClass = NewType("InputDataClass", Any)


logger = logging.getLogger(__name__)
datasets.logging.set_verbosity_error()

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#class NER(TokenClassificationTask):
#    def __init__(self, with_coref, no_ref_idx=0, label_idx=1, coref_idx=-1):
#        # in NER datasets, the last column is usually reserved for NER label
#        # referent index is the second to last column (#marco)
#        self.label_idx = label_idx
#        self.coref_idx = coref_idx
#        self.no_ref_idx = no_ref_idx
#        self.titles = {
#                    'dev':["Pauline","De_la_ville_au_moulin"],
#                    'test':["Pauline"],
#                    'train': ['Jean-Christophe-1','Le_capitaine_Fracasse',
#                                 'Le_diable_au_corps','Le_ventre_de_Paris',
#                                 'Madame_de_Hautefort','Nemoville',
#                                 "Sarrasine",
#                                 "Mademoiselle_Fifi_nouveaux_contes","Douce_Lumiere",
#                                 "Bouvard","Rosalie",
#                                 ]
#                }
#
#    def read_examples_from_folder(self, tokenizer:AutoTokenizer,  data_dir, mode: Union[Split, str]) -> List[CorefInputExample]:
#        if isinstance(mode, Split):
#            mode = mode.value
#        guid_index = 1
#        examples = []
#        for file_path in sorted(glob.glob(os.path.join('.', data_dir+'/*.tsv'))):
#            if mode!="inference" and not any(x in file_path for x in self.titles[mode]) :
#                #les titres correspondant ÃƒÂ  ce split ne comportent pas ce titre
#                #autrement dit, titre non intÃƒÂ©ressant pour ce split
#                #on saute
#                continue
#            with open(file_path, encoding="utf-8") as f:
#                words = []
#                labels = []
#                refs = []
#                book_start = True
#                for line in f:
#                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
#                        if words:
#                            examples.append(CorefInputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, refs=refs, book_start=book_start))
#                            book_start=False
#                            guid_index += 1
#                            words = []
#                            labels = []
#                            refs = []
#                    else:
#                        splits = [elem for elem in line[:-1].split("\t") if elem!='O']
#                        words.append(splits[0])
#                        if len(splits) > 1:
#                            labels.append(splits[self.label_idx] if len(splits)>2 else 'O')
#                            ref = splits[self.coref_idx] if len(splits)>2 else '-'
#                            refs.append(self.no_ref_idx if ref=='-' else ref)
#                        else:
#                            labels.append("O")
#                            refs.append(self.no_ref_idx)
#                if words:
#                    if with_coref:
#                        examples.append(CorefInputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, refs=refs, book_start=book_start))
#                    else:
#                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, book_start=book_start))
#                    book_start=False
#        return examples


def coref_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="If passed, overwrite data cache.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='camembert-base'
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If passed, infer for test set.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="If passed, don't compute metrics.",
    )
    parser.add_argument(
        "--chunk-int",
        type=int,
        default=16,
        help="Chunking overlapping interval.",
    )
    parser.add_argument(
        "--bioes",
        action="store_true",
        help="BIOES scheme instead of BIO.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--coref_pred",
        action="store_true",
        help="Predict coreference as well.",
    )
    args = parser.parse_args()
    # Sanity checks
    if args.data_dir is None:
        raise ValueError("Need data_dir.")

    return args


def main():
    args = parse_args()
    tsv_dir = (args.data_dir[:-1] if args.data_dir[-1]=='/' else args.data_dir)+'_tsv'

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(cpu=False)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.info("Training/evaluation parameters %s", args)

    assert(not (args.inference and args.test)) #inference and test cant both be set True
    # Set seed
    set_seed(42)

    if not os.path.isdir(tsv_dir) or args.overwrite_cache:
        chunk_brat(args.data_dir, tsv_dir,interval=args.chunk_int,max_seq_len=args.max_seq_length,bioes=args.bioes,coref_pred=args.coref_pred)
    
    cmd = "for i in `ls {}/*tsv`; do cut $i -f2 | awk '!a[$0]++'; done | sort | uniq".format(tsv_dir)
    stream = os.popen(cmd)
    output = stream.read()
    label_list = ['O'] + [l for l in output.split('\n') if len(l)>2]
    print("HERE is the list of labels the model is taught to predict :")
    print(label_list)
    print("==========")
    
    label_map: Dict[int, str] = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    titles = {
         'dev':["Jean-Christophe-1","Jean-Christophe-2"],
         'test':["Rosalie"],
         'train': [   
                      "Sarrasine","Pauline",
                      'Le_capitaine_Fracasse',
                      #"elisabeth_Seton",
                      'Le_ventre_de_Paris',
                      'Madame_de_Hautefort','Nemoville',
                      "De_la_ville_au_moulin" ,
                      "Mademoiselle_Fifi_nouveaux_contes-1",
                      "Mademoiselle_Fifi_nouveaux_contes-3",
                      #"export","Bouvard","La_morte_amoureuse",
                      'Le_diable_au_corps','Douce_Lumiere',
                    ]
         }
    titles = {
        'dev':["Pauline","De_la_ville_au_moulin"],
        'test':["Pauline"],
        'train': ['Jean-Christophe-1','Le_capitaine_Fracasse',
                  'Le_diable_au_corps','Le_ventre_de_Paris',
                  'Madame_de_Hautefort','Nemoville',
                  "Sarrasine",
                  "Mademoiselle_Fifi_nouveaux_contes","Douce_Lumiere",
                  "Bouvard","Rosalie",
                  ]
                  }
    print(titles)
    print("==========")
    token_classification_task = NER(with_coref=args.coref_pred, titles=titles, no_ref_idx = args.max_seq_length-1)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(label_list)},
        cache_dir=os.getenv('PYTORCH_TRANSFORMERS_CACHE'),
    )
    bert_module = CamembertForCoreference if args.coref_pred else AutoModelForTokenClassification
    model = bert_module.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=os.getenv('PYTORCH_TRANSFORMERS_CACHE'),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=os.getenv('PYTORCH_TRANSFORMERS_CACHE'),
        use_fast=False,
    )
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    dataset_module = TokenClassificationCorefDataset if args.coref_pred else TokenClassificationDataset
    if not args.test and not args.inference:
        train_dataset = (
            dataset_module(
                token_classification_task=token_classification_task,
                data_dir=tsv_dir,
                tokenizer=tokenizer,
                labels=label_list,
                model_type=config.model_type,
                max_seq_length=args.max_seq_length,
                overwrite_cache=args.overwrite_cache,
                mode=Split.train,
            )
        )
        if args.debug:
            train_dataset = train_dataset[:100]
        eval_dataset = (
            dataset_module(
                token_classification_task=token_classification_task,
                data_dir=tsv_dir,
                tokenizer=tokenizer,
                labels=label_list,
                model_type=config.model_type,
                max_seq_length=args.max_seq_length,
                overwrite_cache=args.overwrite_cache,
                mode=Split.dev,
            )
        )
        if args.debug:
            eval_dataset = eval_dataset[:100]


        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=coref_data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=coref_data_collator, batch_size=args.per_device_eval_batch_size)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


       
        device = accelerator.device
        model.to(device)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = len(train_dataloader)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        total_loss=0
        total_coref_loss=0
        total_ner_loss=0
        min_coref_l = 10**5

        for epoch in range(args.num_train_epochs):
            model.train()
            mini_eval_steps = len(train_dataloader)//4
            adv = epoch/args.num_train_epochs
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                ner_l = outputs.ner_loss if args.coref_pred else outputs.loss
                coref_l = outputs.coref_loss if args.coref_pred else 0
                loss = coref_l + ner_l
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % mini_eval_steps == 0:
                    accelerator.print('loss : {} (ner : {}, coref {})'.format(
                                                np.round(total_loss/mini_eval_steps,3),
                                                np.round(total_ner_loss/mini_eval_steps,3),
                                                np.round(total_coref_loss/mini_eval_steps,3),
                                                ))
                    total_loss=0
                    total_coref_loss=0
                    total_ner_loss=0
                else:
                    total_loss+=loss.item()
                    total_coref_loss+=coref_l.item() if args.coref_pred else 0
                    total_ner_loss+=ner_l.item()

                if completed_steps >= args.max_train_steps:
                    break

            eval_l, _ = run_evaluation(model, eval_dataloader, accelerator, label_list, args.max_seq_length-1, epoch, bioes=args.bioes, muc_only=True, coref_pred=args.coref_pred)

            #check_condition
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                accelerator.print(f"Current loss {eval_l} better than last best loss {min_coref_l}. Saved epoch {epoch} checkpoint.")
                min_coref_l = eval_l
    else :
        #inference OR test
        mode = "inference" if args.inference else Split.test
        test_dataset = (
            TokenClassificationCorefDataset(
                token_classification_task=token_classification_task,
                data_dir=tsv_dir,
                tokenizer=tokenizer,
                labels=label_list,
                model_type=config.model_type,
                max_seq_length=args.max_seq_length,
                overwrite_cache=args.overwrite_cache,
                mode=mode,
            )
        )
        if args.debug:
            test_dataset = test_dataset[:100]

        test_dataloader = DataLoader(test_dataset, collate_fn=coref_data_collator, batch_size=args.per_device_eval_batch_size)
        device = accelerator.device
        model.to(device)
        # Prepare everything with our `accelerator`.
        model, test_dataloader = accelerator.prepare(model, test_dataloader)

        examples = token_classification_task.read_examples_from_folder(tokenizer, tsv_dir, mode)
        if args.debug:
            examples = examples[:100]


        _, (all_ner_preds, all_coref_preds) = run_evaluation(model=model,
                                                            dataloader=test_dataloader,
                                                            accelerator=accelerator,
                                                            label_list=label_list,
                                                            no_ref_idx=args.max_seq_length-1 ,
                                                            epoch_number=-1,
                                                            muc_only=True,
                                                            bioes=args.bioes,
                                                            confidence_ratio=1,
                                                            dont_eval = args.inference)
        books = []
        sentences = []
        for sent_idx, (example, example_ner_pred, example_coref_pred) in enumerate(zip(examples, all_ner_preds, all_coref_preds)):
            w_idx_of_token = []
            sentence = []
            for i,w in enumerate(example.words):
                w_idx_of_token.extend([i]*len(tokenizer.tokenize(w)))
            for i, (w, ner_p, coref_p) in enumerate(zip(example.words, example_ner_pred, example_coref_pred)):
                ref = w_idx_of_token[int(coref_p[1:-1])]+1 if coref_p!='-' else 'O'
                sentence.append((w,ner_p, str(ref)))
            sentences.append(sentence)
            if sent_idx+1== len(examples) or examples[sent_idx+1].book_start:
                books.append(sentences)
                sentences=[]
        if args.test:
            test_titles = sorted(token_classification_task.titles['test'])
        else:
            test_titles = sorted(x.split('/')[-1][:-4] for x in glob.glob(tsv_dir+'/*.tsv'))
        for book_idx,b in enumerate(books):
            write_chunk_and_all_predictions(
                        sentences=b,
                        filename=os.path.join(args.output_dir, f"{test_titles[book_idx]}_chunk_predictions.txt"),
                        chunk_int=args.chunk_int,
                        )
if __name__ == "__main__":
    main()