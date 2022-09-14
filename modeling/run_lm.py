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
import random
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
from utils_coref import Split,TokenClassificationCorefDataset, COREF, CorefInputExample
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
        "--use_cache",
        action="store_true",
        help="If passed, use data cache.",
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
        "--bio",
        action="store_true",
        help="BIO scheme instead of BIOES.",
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
        "--replace_labels",
        type=str,
        default='',
        help="Labels to replace during training, separated by comma and colons,\
        ex: 'SINGER:PER,ACTOR:PER,HIST:TIME",
    )
    parser.add_argument(
        "--ignore_labels",
        type=str,
        default='',
        help="Labels to ignore during trainingi, separated by a comma.",
    )
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
        raise ValueError("Merci de préciser le dossier contenant les données avec l'option --data_dir")
    if args.output_dir is None:
        raise ValueError("Merci de préciser le dossier de sortie avec l'option --output_dir")

    return args


def main():
    args = parse_args()
    tsv_dir = (args.data_dir[:-1] if args.data_dir[-1]=='/' else args.data_dir)+'_tsv'

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,
    )
    logger.info(accelerator.state)

    logger.info("Training/evaluation parameters %s", args)
    
    assert not (args.inference and args.test) #inference and test cant both be set True
    # Set seed
    set_seed(42)
    
    
    cmd = "for i in `ls {}/*tsv`; do cut $i -f2; cut $i -f3 | awk '!a[$0]++'; done | sort | uniq | grep 'B\|I\|E\|S-'"
    if args.inference or args.test :
        labels_fn = os.path.join(args.model_name_or_path,"labels.txt")
        assert os.path.isdir(args.model_name_or_path), "Le dossier {} n'existe pas encore. Si vous importez un modèle de HuggingFace, il faut l'entraîner sur votre tâche avant de l'utiliser pour faire de l'inférence, et donc enlever l'option --inference et/ou --test. Si, en revanche, vous voulez utiliser un modèle déjà entraîné sur cette tâche, il faut indiquer l'adresse de ce dernier dans l'option --model_name_or_path".format(args.model_name_or_path)
        assert os.path.isfile(labels_fn), "Le fichier {0} est introuvable. Ce fichier doit contenir les labels que le modèle a déjà été entraîné à prédire. Si vous disposez des données d'entraînement de celui-ci (sous format tsv), essayez d'exécuter cette commande puis de réessayer : \necho \"O\" > {0};{1} >> {0}\n en remplaçant XXXX par par le nom du dossier, il doit se terminer par \"_tsv\". Sinon, un ré-entraînement est nécessaire.".format(labels_fn,cmd.format('XXXX'))
        with open(labels_fn,"r") as f:
            s = f.read()
            bioes_in_inference = ("E-" in s) or ("S-" in s)
            label_list = s.split('\n')
    
    if not os.path.isdir(tsv_dir) or not args.use_cache:
        chunk_brat(
                args.data_dir,
                tsv_dir,
                should_contain_ents=not args.inference,
                interval=args.chunk_int,
                max_seq_len=args.max_seq_length,
                bioes=bioes_in_inference if args.test or args.inference else not args.bio,
                coref_pred=args.coref_pred,
                labels_to_ignore=args.ignore_labels.split(','),
                labels_to_replace=args.replace_labels.split(',') if args.replace_labels!='' else [],
                )
    
    if not args.inference and not args.test :
        stream = os.popen(cmd.format(tsv_dir))
        output = stream.read()
        label_list = ['O'] + [l for l in output.split('\n') if len(l)>2]
    print("Voici la liste des étiquettes que le modèle est appris à attribuer à chaque mot :")
    print(label_list)
    print("==========")
    
    label_map: Dict[int, str] = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    
    all_titles=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob(args.data_dir+'/*.txt')]
    random.seed(42)
    random.shuffle(all_titles)
    nb_titles = len(all_titles)
    assert args.inference or args.test or nb_titles>1,"Vous êtes en mode entraînement. Or, un seul texte ({}) a été retrouvé dans {}. Ce code est censé fonctionner avec plusieurs oeuvres différentes, pour en isoler une sous-partie et évaluer la qualité du modèle à généraliser pour des oeuvres jamais vues en entraînement.".format(all_titles[0],args.data_dir)
    train_dev_limit = int(nb_titles*.9)
    dev_titles = all_titles[train_dev_limit:]
    train_titles = all_titles[:train_dev_limit]
    test_titles = all_titles
    if not args.test and not args.inference:
        print("Oeuvres utilsées pour l'entraînement :")
        for t in train_titles:
            print(t)
        print('----------')
        print("Oeuvres utilsées pour l'évaluation interne :")
        for t in dev_titles:
            print(t)
        print("==========")
    titles = {'train':train_titles, 'dev':dev_titles, 'test':test_titles}
    if not args.coref_pred:
        token_classification_task = NER(with_coref=args.coref_pred, titles=titles, no_ref_idx = args.max_seq_length-1)
    else:
        token_classification_task = COREF(with_coref=args.coref_pred, titles=titles, no_ref_idx = args.max_seq_length-1)

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
                overwrite_cache= not args.use_cache,
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
                overwrite_cache=not args.use_cache,
                mode=Split.dev,
            )
        )
        if args.debug:
            eval_dataset = eval_dataset[:100]
        
        assert len(train_dataset)>0, "Vous utilisez un cache vide. Essayez d'enlever l'option --use_cache. Si cela ne marche pas, vérifiez que le dossier {} contient bien vos fichiers txt.".format(args.data_dir)

        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

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

            eval_l, _ = run_evaluation(model, eval_dataloader, accelerator, label_list, args.max_seq_length-1, epoch, bioes=not args.bio, muc_only=True, coref_pred=args.coref_pred)

            #check_condition
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                with open(os.path.join(args.output_dir, "labels.txt"), "w+") as f:
                    f.write("\n".join(label_list))
                accelerator.print(f"Current loss {eval_l} better than last best loss {min_coref_l}. Saved epoch {epoch} checkpoint.")
                min_coref_l = eval_l
    else :
        #inference OR test
        mode = Split.inference if args.inference else Split.test
        test_dataset = (
            dataset_module(
                token_classification_task=token_classification_task,
                data_dir=tsv_dir,
                tokenizer=tokenizer,
                labels=label_list,
                model_type=config.model_type,
                max_seq_length=args.max_seq_length,
                overwrite_cache=not args.use_cache,
                mode=mode,
            )
        )
        if args.debug:
            test_dataset = test_dataset[:100]
        
        if args.bio and bioes_in_inference:
            print("L'argument --bio a été négligé parce le modèle a été entraîné sur le format bioes.")
        elif not args.bio and not bioes_in_inference:
            print("Le modèle a été entraîne sur le format bio, mais vous n'avez pas entré l'argument --bio. Le mode bio a été automatiquement défini")
        test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)
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
                                                            bioes=bioes_in_inference,
                                                            confidence_ratio=1,
                                                            dont_eval = args.inference,
                                                            coref_pred = args.coref_pred)
        books = []
        sentences = []
        for sent_idx, (example, example_ner_pred, example_coref_pred) in enumerate(zip(examples, all_ner_preds, all_coref_preds)):
            w_idx_of_token = []
            sentence = []
            for i,w in enumerate(example.words):
                w_idx_of_token.extend([i]*len(tokenizer.tokenize(w)))
            for i, (w, ner_p, coref_p) in enumerate(zip(example.words, example_ner_pred, example_coref_pred)):
                if coref_p!='-' and int(coref_p[1:-1])<len(w_idx_of_token):
                    ref = w_idx_of_token[int(coref_p[1:-1])]+1
                else:
                    ref = 'O'
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
                        filename=os.path.join(args.output_dir, f"{test_titles[book_idx]}.tsv"),
                        chunk_int=args.chunk_int,
                        bioes=bioes_in_inference,
                        text_filename=os.path.join(args.data_dir, f"{test_titles[book_idx]}.txt")
                        )
if __name__ == "__main__":
    main()
