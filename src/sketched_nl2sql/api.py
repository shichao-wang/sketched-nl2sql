""" defines entry point for sketched nl2sql """
import logging
import sqlite3
from os import path
from typing import List

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import torchnlp.utils
from sketched_nl2sql import scorer
from sketched_nl2sql.dataset import collate_fn, Example, WikisqlDataset
from sketched_nl2sql.engine import Engine
from torchnlp.config import Config
from torchnlp.vocab import Vocab

logger = logging.getLogger(__name__)


def train(data_path: str, save_path: str, config: Config, resume: bool = True):
    """ training method for sketched nl2sql"""
    torchnlp.utils.set_random_seed(config.get("seed", None))

    if resume:
        logger.info("Try Loading Checkpoint")
        runner = Engine.from_checkpoint(config, save_path)
        logger.info("Loaded from ckpt")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.get("pretrained_model_name"))
        vocab = Vocab.from_pretrained_tokenizer(
            tokenizer, pad_token="[PAD]", unk_token="[UNK]", sep_token="[SEP]", cls_token="[CLS]"
        )
        runner = Engine(config, tokenizer, vocab)

    train_set = WikisqlDataset(data_path, "train", tokenize=runner.tokenizer.tokenize)
    train_loader = DataLoader(
        train_set, batch_size=config.get("batch_size"), shuffle=True, collate_fn=list, pin_memory=True, num_workers=4
    )
    batchifier = collate_fn(runner.vocab, runner.device)

    train_tqdm = tqdm(train_loader)

    total_gold_examples: List[Example] = []
    total_pred_examples: List[Example] = []
    try:
        for epoch_id in enumerate(range(config.get("max_epoch", 10)), start=1):

            for examples in train_tqdm:  # type: List[Example]
                inputs, targets = batchifier(examples)
                loss = runner.feed(inputs, targets)
                # evaluate
                gold_examples = examples
                total_gold_examples.extend(gold_examples)
                pred_queries = runner.predict(inputs)
                pred_examples = [
                    Example(example.question_tokens, example.header, pred_query)
                    for example, pred_query in zip(examples, pred_queries)
                ]
                total_pred_examples.extend(pred_examples)

                exact_acc = sum(
                    [
                        scorer.exact_accuracy(pred_example, gold_example)
                        for pred_example, gold_example in zip(pred_examples, gold_examples)
                    ]
                ) / len(gold_examples)
                with sqlite3.connect(path.join(data_path, "train.db")) as conn:
                    logic_acc = sum(
                        [
                            scorer.logical_accuracy(
                                runner.build_query_string(pred_query), runner.build_query_string(gold_query), conn
                            )
                            for pred_query, gold_query in zip(pred_examples, gold_examples,)
                        ]
                    ) / len(gold_examples)
                train_tqdm.set_postfix_str(f"loss: {loss}  exact_acc: {exact_acc} logical_acc: {logic_acc}")

            exact_acc = sum(
                [
                    scorer.exact_accuracy(pred_example, gold_example)
                    for pred_example, gold_example in tqdm(
                        zip(total_pred_examples, total_gold_examples), desc="Epoch Exact Accuracy"
                    )
                ]
            ) / len(total_gold_examples)
            with sqlite3.connect(path.join(data_path, "train.db")) as conn:
                logic_acc = sum(
                    [
                        scorer.logical_accuracy(
                            runner.build_query_string(pred_query), runner.build_query_string(gold_query), conn
                        )
                        for pred_query, gold_query in tqdm(
                            zip(total_pred_examples, total_gold_examples), desc="Epoch Logical Accuracy"
                        )
                    ]
                ) / len(total_gold_examples)
            logger.info(f"epoch: {epoch_id} exact_acc: {exact_acc} logic_acc: {logic_acc}")

    except BaseException as e:
        runner.save_checkpoint(save_path)
        raise e
