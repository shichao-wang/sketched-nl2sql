from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import chain
from operator import attrgetter, itemgetter
from os import path
from typing import Callable, Dict, List, NamedTuple

import torch
from more_itertools import collapse, flatten
from nltk import TweetTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data import batchify
from data.vocab import Vocab
from sketched_nl2sql import utils

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


class SketchedQuery(NamedTuple):
    """ data container for query object """

    agg_id: int
    col_id: int
    conds: List[Cond] = None

    class Cond(NamedTuple):
        """ conditions """

        col_id: int
        op_id: int
        value_beg: int
        value_end: int


class WikiSqlExample(NamedTuple):
    """ data container for WikiSql dataset s"""

    question_tokens: List[str]
    table_id: str
    query: SketchedQuery = None


def remove_unicode(line: str):
    """ remove """
    return re.sub(r"(?P<unicode>\\u\w{4})(?P<digital>\d+)", r"\g<unicode> \g<digital>", line)


def load_wikisql(wiki_sql_data: str, tables_file: str, tokenize_func: Callable[[str], List[str]]):
    """ main load function """
    headers = {}
    with open(tables_file) as fp:
        for data in map(json.loads, fp):
            table_id: str = data["id"]
            tokens = [tokenize_func(h) for h in data["header"]]
            types = [t for t in data["types"]]
            headers[table_id] = {"tokens_list": tokens, "types": types}
    # read tables
    examples = []
    with open(wiki_sql_data) as fp:
        for table_id, question, sql in tqdm(
            map(itemgetter("table_id", "question", "sql"), map(json.loads, map(remove_unicode, fp))),
            desc="Reading Dataset",
        ):
            question_tokens = tokenize_func(question)
            conds = []
            for cond in sql["conds"]:
                col_id, op_id, value = cond
                value_tokens = tokenize_func(str(value))
                start_pos, end_pos = utils.find_value_position(question_tokens, value_tokens)
                conds.append(SketchedQuery.Cond(col_id, op_id, start_pos, end_pos))
            query = SketchedQuery(sql["agg"], sql["sel"], conds)
            example = WikiSqlExample(question_tokens, table_id, query)
            examples.append(example)
    return headers, examples


class WikisqlDataset(Dataset):
    """ wikisql dataset """

    def __init__(self, data_path: str, split: str, tokenizer=None, vocab: Vocab = None):
        # load data
        data_file = path.join(data_path, split) + ".jsonl"
        tables_file = path.join(data_path, split) + ".tables.jsonl"

        # build vocabulary
        special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            # "bos_token": "[BOS]",
            # "eos_token": "[EOS]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]",
        }
        tokenizer = tokenizer or TweetTokenizer(preserve_case=False)
        if isinstance(tokenizer, PreTrainedTokenizer):  # tokenizer that contains vocab
            self.vocab = vocab or Vocab.from_pretrained_tokenizer(tokenizer, **special_tokens)
            headers, examples = load_wikisql(data_file, tables_file, tokenizer.tokenize)
        else:
            headers, examples = load_wikisql(data_file, tables_file, tokenizer.tokenize)
            question_tokens = flatten(map(attrgetter("question_tokens"), examples))
            header_tokens = collapse(map(itemgetter("tokens_list"), headers.values()), str)
            counter = Counter(chain(question_tokens, header_tokens))
            self.vocab = vocab or Vocab.from_counter(counter, **special_tokens)
        # header_tokens = flatten(map(itemgetter("types"), headers.values()))
        # self.type_vocab = Vocab.from_iterable(header_tokens)

        # process data, process table first then examples
        headers_tensor: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        for table_id, header in headers.items():

            headers_tokens_tensor_list = []
            headers_tokens_mask_list = []
            for tokens in header["tokens_list"]:
                headers_tokens_tensor_list.extend(self.vocab.map2index(tokens) + [self.vocab.sep_index])
                headers_tokens_mask_list.extend([1] * len(tokens) + [-2])
            headers_tensor[table_id]["headers_tokens"] = headers_tokens_tensor_list
            headers_tensor[table_id]["headers_tokens_mask"] = headers_tokens_mask_list

            # header_types = self.type_vocab.map2index(header["types"])
            # headers_tensor[table_id]["header_types"] = header_types

        self.tensor_examples = []
        example: WikiSqlExample
        for example in tqdm(examples, "Processing examples"):
            header_tensor = headers_tensor[example.table_id]
            input_tokens = torch.as_tensor(
                [self.vocab.cls_index] + header_tensor["headers_tokens"]
                # + [self.vocab.sep_index]
                + self.vocab.map2index(example.question_tokens),
                dtype=torch.long,
            )
            input_segment = torch.as_tensor(
                [-1] + header_tensor["headers_tokens_mask"]
                # + [-2]
                + [2] * len(example.question_tokens),
                dtype=torch.long,
            )
            input_data = (input_tokens, input_segment)
            if example.query.conds:
                conditions = torch.as_tensor(
                    [[cond.col_id, cond.op_id, cond.value_beg, cond.value_end] for cond in example.query.conds]
                ).unbind(1)
            else:
                conditions = torch.as_tensor(([], [], [], []), dtype=torch.long)
            target_data = (
                torch.as_tensor(example.query.agg_id),
                torch.as_tensor(example.query.col_id),
                torch.as_tensor(len(example.query.conds)),
                *conditions,
            )
            self.tensor_examples.append((input_data, target_data))

    def __getitem__(self, index):
        return self.tensor_examples[index]

    def __len__(self):
        return len(self.tensor_examples)

    batch_fn = batchify.tuples(
        batchify.tuples(batchify.sequences(), batchify.sequences()),
        batchify.tuples(
            batchify.arrays(),
            batchify.arrays(),
            batchify.arrays(),
            batchify.skip(),
            batchify.skip(),
            batchify.skip(),
            batchify.skip(),
        ),
    )


# def batch_fn(examples: List[Tuple[Input, SketchedQuery]]):
#     question_tokens_list, header_tokens_list, header_type_list = list(zip(*examples))
#     padded_question_tokens = rnn.pad_sequence(question_tokens_list, True,)
