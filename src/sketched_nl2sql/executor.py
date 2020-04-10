""" executor """
import logging
import operator
import sqlite3
from os import path
from typing import List

import numpy
from pytorch_transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import torchnlp.utils
from sketched_nl2sql.dataset import collate_fn, Example, WikisqlDataset
from sketched_nl2sql.engine import Engine
from torchnlp.config import Config
from torchnlp.vocab import Vocab

logger = logging.getLogger(__name__)

__all__ = ["train"]


def train(data_path: str, config: Config, checkpoint_path: str, *, resume: bool):
    """ train process """
    # loading engine
    torchnlp.utils.set_random_seed(config.get("seed", None))
    if resume:
        engine = Engine.from_checkpoint(config, checkpoint_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.get("pretrained_model_name"))
        vocab = Vocab.from_pretrained_tokenizer(
            tokenizer, pad_token="[PAD]", unk_token="[UNK]", sep_token="[SEP]", cls_token="[CLS]"
        )
        engine = Engine(config, tokenizer, vocab)

    # preparing data
    batchifier = collate_fn(engine.vocab, engine.device)

    train_set = WikisqlDataset(data_path, "train", tokenize=engine.tokenizer.tokenize)
    train_loader = DataLoader(
        train_set, batch_size=config.get("batch_size"), shuffle=True, collate_fn=list, num_workers=4
    )
    dev_set = WikisqlDataset(data_path, "dev", engine.tokenizer.tokenize)
    dev_loader = DataLoader(dev_set, batch_size=config.get("batch_size"), collate_fn=list)

    test_set = WikisqlDataset(data_path, "test", engine.tokenizer.tokenize)
    test_loader = DataLoader(dev_set, batch_size=config.get("batch_size"), collate_fn=list)

    def evaluate(preds: List[Example], golds: List[Example], conn: sqlite3.Connection, log_false: bool = False):
        """ evaluate """
        query_getter = operator.attrgetter("query")
        logic_acc = numpy.mean([eq for eq in map(operator.eq, map(query_getter, preds), map(query_getter, golds))])
        exec_results: List[bool] = []
        for pred_qs, gold_qs in zip(
            map(engine.build_query_string, preds), map(engine.build_query_string, golds)
        ):  # type: str, str
            eq = False
            try:
                pred = conn.execute(pred_qs).fetchall()
                gold = conn.execute(gold_qs).fetchall()
                eq = pred == gold
                if not eq and log_false:
                    logger.info(f"False in results:\ngold: {gold_qs}\npred: {pred_qs} ")
            except sqlite3.Error:
                if log_false:
                    logger.info(f"False in execution:\ngold: {gold_qs}\npred: {pred_qs} ")
            exec_results.append(eq)

        exec_acc = numpy.mean(exec_results)
        return logic_acc, exec_acc

    def train_epoch(epoch_num: int):
        """ train batch """
        train_tqdm = tqdm(train_loader)
        for gold_examples in train_tqdm:  # type: List[Example]
            inputs, targets = batchifier(gold_examples)
            loss = engine.feed(inputs, targets)

            pred_queries = engine.predict(inputs)
            pred_examples = [
                Example(example.question_tokens, example.header, pred_query)
                for example, pred_query in zip(gold_examples, pred_queries)
            ]
            with sqlite3.connect(path.join(data_path, "train.db")) as conn:
                logic_acc, exec_acc = evaluate(pred_examples, gold_examples, conn)

            train_tqdm.set_postfix({"loss": loss, "logic_acc": logic_acc, "exec_acc": exec_acc})
        engine.save_checkpoint(checkpoint_path)

    def evaluate_epoch(loader: DataLoader, label: str, db_file: str):
        """ evaluate on one set """
        epoch_pred_examples, epoch_gold_examples = [], []
        for gold_examples in tqdm(loader, desc=label):
            inputs, targets = batchifier(gold_examples)
            pred_queries = engine.predict(inputs)
            pred_examples = [
                Example(example.question_tokens, example.header, pred_query)
                for example, pred_query in zip(gold_examples, pred_queries)
            ]
            epoch_pred_examples.extend(pred_examples)
            epoch_gold_examples.extend(gold_examples)

        with sqlite3.connect(path.join(data_path, db_file)) as conn:
            logic_acc, exec_acc = evaluate(epoch_pred_examples, epoch_gold_examples, conn)

        return logic_acc, exec_acc

    for epoch in range(1, config.get("max_epoch", 10) + 1):
        try:
            train_epoch(epoch)
            train_logic_acc, train_exec_acc = evaluate_epoch(train_loader, "Evaluating on Training Set", "train.db")
            logger.info(f"Epoch: {epoch} Training Logic Acc: {train_logic_acc} Training Exec Acc: {train_exec_acc}")
            dev_logic_acc, dev_exec_acc = evaluate_epoch(dev_loader, "Evaluating on Development Set", "dev.db")
            logger.info(f"Epoch: {epoch} Development Logic Acc: {dev_logic_acc} Development Exec Acc: {dev_exec_acc}")
            test_logic_acc, test_exec_acc = evaluate_epoch(test_loader, "Evaluating on Testing Set", "test.db")
            logger.info(f"Epoch: {epoch} Testing Logic Acc: {test_logic_acc} Testing Exec Acc: {test_exec_acc}")
        except BaseException:
            engine.save_checkpoint(checkpoint_path)
            logger.info("Saved on Exception")
            raise


class Executor:
    """ Executor is a wrapper class for Engine.
    Since engine controls between data(batch) and model, Executor controls between dataset and engine.

    1. Support Parallel.
    """

    def __init__(self, config: Config, *, resume: bool):
        torchnlp.utils.set_random_seed(config.get("seed", None))

    def train(self, data_loader: DataLoader):
        """ train on data loader"""
