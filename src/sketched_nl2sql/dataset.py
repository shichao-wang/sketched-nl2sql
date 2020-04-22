""" Dataset module """
import json
import re
from operator import itemgetter
from os import path
from typing import Callable, Dict, List, NamedTuple, Set

import torch
from torch import Tensor
from torch.nn.utils import rnn
from torch.utils.data import Dataset
from tqdm import tqdm

from torchnlp.vocab import Vocab

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


class Cond(NamedTuple):
    col_id: int
    op_id: int
    value_start: int
    value_end: int


class Query(NamedTuple):
    """ target """

    sel_col_id: int
    agg_id: int
    conds: Set[Cond]


class Header(NamedTuple):
    """ container for header """

    table_id: str
    column_tokens: List[List[str]]
    types: List[str]

    def __len__(self):
        return len(self.column_tokens)


class Example(NamedTuple):
    """ data container for WikiSql dataset s"""

    question_tokens: List[str]
    header: Header
    query: Query


def remove_unicode(line: str):
    """ remove """
    return re.sub(
        r"(?P<unicode>\\u\w{4})(?P<digital>\d+)",
        r"\g<unicode> \g<digital>",
        line,
    )


def load_wikisql(
    wiki_sql_data: str, tables_file: str, tokenize: Callable[[str], List[str]]
):
    """ main load function """
    named_headers: Dict[str, Header] = {}
    with open(tables_file) as fp:
        for data in tqdm(map(json.loads, fp), desc="Reading Tables File"):
            tid: str = data["id"]
            named_headers[tid] = Header(
                tid, [tokenize(col) for col in data["header"]], data["types"],
            )
    examples = []
    with open(wiki_sql_data) as fp:
        for table_id, question, sql in tqdm(
            map(
                itemgetter("table_id", "question", "sql"),
                map(json.loads, map(remove_unicode, fp)),
            ),
            desc="Reading Dataset",
        ):  # type: str, str, Dict
            question_tokens = tokenize(question)
            conds = set()
            for cond in sql["conds"]:
                col_id, op_id, value = cond
                value_tokens = tokenize(str(value))
                start, end = find_value_position(question_tokens, value_tokens)
                conds.add(Cond(col_id, op_id, start, end))

            query = Query(sql["sel"], sql["agg"], conds)
            example = Example(question_tokens, named_headers[table_id], query)
            examples.append(example)

    return examples


class WikisqlDataset(Dataset):
    """ wikisql dataset """

    def __init__(
        self, data_path: str, split: str, tokenize: Callable[[str], List[str]]
    ):
        # load data
        data_file = path.join(data_path, split) + ".jsonl"
        tables_file = path.join(data_path, split) + ".tables.jsonl"
        self.examples = load_wikisql(data_file, tables_file, tokenize)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def __len__(self):
        return len(self.examples)


def collate_fn(vocab: Vocab, device: torch.device):
    """ wikisql collate function """

    def _collate(examples: List[Example]):
        questions_tokens, headers, queries = zip(
            *examples
        )  # type: List[List[str]], List[Header], List[Query]
        questions_tensor = rnn.pad_sequence(
            [
                torch.as_tensor(
                    vocab.map2index(tokens), dtype=torch.long, device=device
                )
                for tokens in questions_tokens
            ],
            batch_first=True,
            padding_value=vocab.pad_index,
        )  # type: Tensor
        headers_tensor = rnn.pad_sequence(
            [
                torch.as_tensor(
                    vocab.map2index(tokens), dtype=torch.long, device=device
                )
                for header in headers
                for tokens in header.column_tokens
            ],
            batch_first=True,
            padding_value=vocab.pad_index,
        )
        header_lengths = [len(header) for header in headers]
        sel_col_ids, agg_ids, conds_batch = zip(*queries)
        cond_col_ids, cond_col_ops, cond_starts, cond_ends = [], [], [], []
        for conds in conds_batch:
            cond_col_ids.append(
                torch.as_tensor(
                    [cond.col_id for cond in conds],
                    dtype=torch.long,
                    device=device,
                )
            )
            cond_col_ops.append(
                torch.as_tensor(
                    [cond.op_id for cond in conds],
                    dtype=torch.long,
                    device=device,
                )
            )
            cond_starts.append(
                torch.as_tensor(
                    [cond.value_start for cond in conds],
                    dtype=torch.long,
                    device=device,
                )
            )
            cond_ends.append(
                torch.as_tensor(
                    [cond.value_end for cond in conds],
                    dtype=torch.long,
                    device=device,
                )
            )
        return (
            (questions_tensor, headers_tensor, header_lengths),
            (
                torch.as_tensor(sel_col_ids, dtype=torch.long, device=device),
                torch.as_tensor(agg_ids, dtype=torch.long, device=device),
                torch.as_tensor(
                    [len(conds) for conds in conds_batch],
                    dtype=torch.long,
                    device=device,
                ),
                cond_col_ids,
                cond_col_ops,
                cond_starts,
                cond_ends,
            ),
        )

    return _collate


def similar(token1: str, token2: str):
    token1, token2 = token1.lower(), token2.lower()
    return token1 in token2 or token2 in token1


def find_value_position(question_tokens: List[str], value_tokens: List[str]):
    length = len(value_tokens)
    for index in (
        index
        for index, t in enumerate(question_tokens)
        if similar(t, value_tokens[0])
    ):
        if all(
            similar(q, v)
            for q, v in zip(
                question_tokens[index : index + length], value_tokens
            )
        ):
            return index, index + length - 1
    raise ValueError(question_tokens + ["\n"] + value_tokens)
