import json
import re
from operator import itemgetter
from os import path
from typing import List, NamedTuple

from torch.utils.data import Dataset
from tqdm import tqdm

from torchnlp import batchify

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


class Cond(NamedTuple):
    """ conditions """

    col_id: int
    op_id: int
    value: str


class SketchedQuery(NamedTuple):
    """ data container for query object """

    agg_id: int
    col_id: int
    conds: List[Cond] = []


class WikiSqlExample(NamedTuple):
    """ data container for WikiSql dataset s"""

    question_tokens: str
    table_id: str
    query: SketchedQuery = None


def remove_unicode(line: str):
    """ remove """
    return re.sub(r"(?P<unicode>\\u\w{4})(?P<digital>\d+)", r"\g<unicode> \g<digital>", line)


def load_wikisql(wiki_sql_data: str, tables_file: str):
    """ main load function """
    named_headers = {}
    with open(tables_file) as fp:
        for data in map(json.loads, fp):
            table_id: str = data["id"]
            headers = {"headers": [h for h in data["header"]], "types": data["types"]}  # avoid being batched
            named_headers[table_id] = headers
    # read tables
    examples = []
    with open(wiki_sql_data) as fp:
        for table_id, question, sql in tqdm(
            map(itemgetter("table_id", "question", "sql"), map(json.loads, map(remove_unicode, fp))),
            desc="Reading Dataset",
        ):
            # question_tokens = tokenize(question)
            conds = []
            for cond in sql["conds"]:
                col_id, op_id, value = cond
                # value_tokens = tokenize_func(str(value))
                # start_pos, end_pos = find_value_position(question, value_tokens)
                conds.append(Cond(col_id, op_id, str(value)))
            query = SketchedQuery(sql["agg"], sql["sel"], conds)
            example = WikiSqlExample(question, table_id, query)
            examples.append(example)
    return named_headers, examples


class WikisqlDataset(Dataset):
    """ wikisql dataset """

    def __init__(self, data_path: str, split: str):
        # load data
        data_file = path.join(data_path, split) + ".jsonl"
        tables_file = path.join(data_path, split) + ".tables.jsonl"
        self.headers, self.examples = load_wikisql(data_file, tables_file)

        # # build vocabulary
        # special_tokens = {
        #     "pad_token": "[PAD]",
        #     "unk_token": "[UNK]",
        #     "sep_token": "[SEP]",
        #     "cls_token": "[CLS]",
        # }

        # # header_tokens = flatten(map(itemgetter("types"), headers.values()))
        # # self.type_vocab = Vocab.from_iterable(header_tokens)
        #
        # # process data, process table first then examples
        # headers_tensor: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        # for table_id, header in headers.items():
        #     headers_tokens_tensor_list = []
        #     headers_tokens_mask_list = []
        #     for tokens in header["tokens_list"]:
        #         headers_tokens_tensor_list.extend(self.vocab.map2index(tokens) + [self.vocab.sep_index])
        #         headers_tokens_mask_list.extend([1] * len(tokens) + [-2])
        #     headers_tensor[table_id]["headers_tokens"] = headers_tokens_tensor_list
        #     headers_tensor[table_id]["headers_tokens_mask"] = headers_tokens_mask_list
        #
        #     # header_types = self.type_vocab.map2index(header["types"])
        #     # headers_tensor[table_id]["header_types"] = header_types
        #
        # self.tensor_examples = []
        # example: WikiSqlExample
        # for example in tqdm(examples, "Processing examples"):
        #     header_tensor = headers_tensor[example.table_id]
        #     input_tokens = torch.as_tensor(
        #         [self.vocab.cls_index] + header_tensor["headers_tokens"]
        #         # + [self.vocab.sep_index]
        #         + self.vocab.map2index(example.question),
        #         dtype=torch.long,
        #     )
        #     input_segment = torch.as_tensor(
        #         [-1] + header_tensor["headers_tokens_mask"]
        #         # + [-2]
        #         + [2] * len(example.question),
        #         dtype=torch.long,
        #     )
        #     input_data = (input_tokens, input_segment)
        #     if example.query.conds:
        #         conditions = torch.as_tensor(
        #             [[cond.col_id, cond.op_id, cond.value_beg, cond.value_end] for cond in example.query.conds]
        #         ).unbind(1)
        #     else:
        #         conditions = torch.as_tensor(([], [], [], []), dtype=torch.long)
        #     target_data = (
        #         torch.as_tensor(example.query.agg_id),
        #         torch.as_tensor(example.query.col_id),
        #         torch.as_tensor(len(example.query.conds)),
        #         *conditions,
        #     )
        #     self.tensor_examples.append((input_data, target_data))

    def __getitem__(self, index):
        example = self.examples[index]
        headers = self.headers[example.table_id]

        return (
            (example.question_tokens, example.table_id, headers["headers"]),
            (
                example.query.col_id,
                example.query.agg_id,
                len(example.query.conds),
                [cond.col_id for cond in example.query.conds],
                [cond.op_id for cond in example.query.conds],
                [cond.value for cond in example.query.conds],
            ),
        )

    def __len__(self):
        return len(self.examples)


batchifier = batchify.tuples(
    batchify.tuples(batchify.skip(), batchify.skip(), batchify.skip()),
    batchify.tuples(
        batchify.skip(), batchify.skip(), batchify.skip(), batchify.skip(), batchify.skip(), batchify.skip(),
    ),
)


def similar(token1: str, token2: str):
    token1, token2 = token1.lower(), token2.lower()
    return token1 in token2 or token2 in token1


def find_value_position(question_tokens: List[str], value_tokens: List[str]):
    length = len(value_tokens)
    for index in (index for index, t in enumerate(question_tokens) if similar(t, value_tokens[0])):
        if all(similar(q, v) for q, v in zip(question_tokens[index : index + length], value_tokens)):
            return index, index + length - 1
    raise ValueError(question_tokens + "\n" + value_tokens)
