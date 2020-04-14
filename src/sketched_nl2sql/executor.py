""" executor """
import logging
import operator
import sqlite3
from os import path
from typing import List

import numpy
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import torchnlp.utils
from sketched_nl2sql.dataset import Example, WikisqlDataset, collate_fn
from sketched_nl2sql.engine import Engine
from torchnlp.config import Config
from torchnlp.vocab import Vocab

logger = logging.getLogger(__name__)

__all__ = ["train"]


def train(
    data_path: str, config: Config, checkpoint_path: str, *, resume: bool
):
    """ train process """
    # loading engine
    if "seed" in config:
        torchnlp.utils.set_random_seed(config.get_int("seed"))
    if resume:
        engine = Engine.from_checkpoint(config, checkpoint_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.get("pretrained_model_name")
        )
        vocab = Vocab.from_pretrained_tokenizer(
            tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            cls_token="[CLS]",
        )
        engine = Engine(config, tokenizer, vocab)

    # preparing data
    batchifier = collate_fn(engine.vocab, engine.device)

    train_set = WikisqlDataset(
        data_path, "train", tokenize=engine.tokenizer.tokenize
    )
    dev_set = WikisqlDataset(
        data_path, "dev", tokenize=engine.tokenizer.tokenize
    )
    test_set = WikisqlDataset(data_path, "test", engine.tokenizer.tokenize)
    train_loader = DataLoader(
        train_set,
        batch_size=int(config.get("batch_size")),
        shuffle=True,
        collate_fn=list,
        num_workers=8,
    )

    def evaluate(
        preds: List[Example],
        golds: List[Example],
        conn: sqlite3.Connection,
        log_false: bool = False,
    ):
        """ evaluate """
        query_getter = operator.attrgetter("query")
        logic_acc = numpy.mean(
            list(
                map(
                    operator.eq,
                    map(query_getter, preds),
                    map(query_getter, golds),
                )
            )
        )
        exec_results: List[bool] = []
        for pred_qs, gold_qs in zip(
            map(engine.build_query_string, preds),
            map(engine.build_query_string, golds),
        ):  # type: str, str
            is_equal = False
            try:
                pred = conn.execute(pred_qs).fetchall()
                gold = conn.execute(gold_qs).fetchall()
                is_equal = pred == gold
                if not is_equal and log_false:
                    logger.info(
                        f"False in results: gold: {gold_qs} pred: {pred_qs} "
                    )
            except sqlite3.Error:
                if log_false:
                    logger.info(
                        f"False in execution: gold: {gold_qs} pred: {pred_qs} "
                    )
            exec_results.append(is_equal)

        exec_acc = numpy.mean(exec_results)
        return logic_acc, exec_acc

    def train_epoch(epoch_num: int):
        """ train batch """
        del epoch_num
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
                logic_acc, exec_acc = evaluate(
                    pred_examples, gold_examples, conn
                )

            train_tqdm.set_postfix(
                {"loss": loss, "logic_acc": logic_acc, "exec_acc": exec_acc}
            )
        engine.save_checkpoint(checkpoint_path)

    def evaluate_epoch(dataset: Dataset, label: str, db_file: str):
        """ evaluate on one set """
        loader = DataLoader(
            dataset, batch_size=int(config.get("batch_size")), collate_fn=list
        )
        epoch_pred_examples, epoch_gold_examples = [], []
        for gold_examples in tqdm(loader, desc=label):
            inputs, _ = batchifier(gold_examples)
            pred_queries = engine.predict(inputs)
            pred_examples = [
                Example(example.question_tokens, example.header, pred_query)
                for example, pred_query in zip(gold_examples, pred_queries)
            ]
            epoch_pred_examples.extend(pred_examples)
            epoch_gold_examples.extend(gold_examples)

        with sqlite3.connect(path.join(data_path, db_file)) as conn:
            logic_acc, exec_acc = evaluate(
                epoch_pred_examples, epoch_gold_examples, conn
            )

        return logic_acc, exec_acc

    for epoch in range(1, int(config.get("max_epoch", 10)) + 1):
        try:
            train_logic_acc, train_exec_acc = evaluate_epoch(
                train_set, "Evaluating on Training Set", "train.db"
            )
            logger.info(
                f"Epoch: {epoch}\t"
                f"Training Logic Acc: {train_logic_acc}\t"
                f"Exec Acc: {train_exec_acc}"
            )
            dev_logic_acc, dev_exec_acc = evaluate_epoch(
                dev_set, "Evaluating on Development Set", "dev.db"
            )
            logger.info(
                f"Epoch: {epoch}\t"
                f"Development Logic Acc: {dev_logic_acc}\t"
                f"Exec Acc: {dev_exec_acc}"
            )
            test_logic_acc, test_exec_acc = evaluate_epoch(
                test_set, "Evaluating on Testing Set", "test.db"
            )
            logger.info(
                f"Epoch: {epoch} Testing Logic Acc: {test_logic_acc}\t"
                f"Exec Acc: {test_exec_acc}"
            )
            train_epoch(epoch)
        except BaseException:
            engine.save_checkpoint(checkpoint_path)
            logger.info("Saved on Exception")
            raise


# class Executor:
#     """ Executor is a wrapper class for Engine.
#     Since engine controls between data(batch) and model,
#     Executor controls between dataset and engine.

#     1. Support Parallel.
#     """

#     def __init__(self, config: Config, *, resume: bool):
#         seed = config.get("seed", None)
#         if seed:
#             torchnlp.utils.set_random_seed(int(seed))
#         self.resume = resume
#         # resume should not be a state in class
#            but variable indicating how to run

#     def train(self, data_loader: DataLoader):
#         """ train on data loader"""
