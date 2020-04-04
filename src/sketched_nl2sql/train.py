from __future__ import annotations

from argparse import ArgumentParser, Namespace

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.dataset import WikisqlDataset
from sketched_nl2sql import utils
from sketched_nl2sql.engine import Engine

string_args = [
    "--data_path",
    "/Users/chaoftic/Public/数据集/WikiSQL",
    "--seed",
    "1234",
    "--pretrained_model_name",
    "/Users/chaoftic/Public/bert-base-uncased",
    "--hidden_dim",
    "300",
]


def parse_args():
    """ arg parser """
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pretrained_model_name", type=str)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_aggs", type=int)
    parser.set_defaults(num_aggs=6)
    return parser.parse_args(string_args)


def main(args: Namespace):
    """ main """
    seed = getattr(args, "seed", None)
    if seed:
        utils.set_random_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    train_set = WikisqlDataset(args.data_path, "train", tokenizer)
    train_loader = DataLoader(train_set, batch_size=8, collate_fn=WikisqlDataset.batch_fn)

    engine = Engine(args, train_set.vocab)

    train_tqdm = tqdm(train_loader)
    try:
        for batch_data in train_tqdm:
            loss = engine.feed(batch_data)
            train_tqdm.set_description(f"loss: {loss}")
    except Exception as e:
        engine.save_checkpoint("some path")


if __name__ == "__main__":
    main(parse_args())
