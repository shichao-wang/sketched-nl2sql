""" defines entry point for sketched nl2sql """

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import torchnlp.utils
from sketched_nl2sql.dataset import batchifier, WikisqlDataset
from sketched_nl2sql.engine import Engine
from torchnlp.config import Config
from torchnlp.vocab import Vocab


def train(data_path: str, save_path: str, config: Config, seed: int = None):
    """ training method for sketched nl2sql"""
    if seed:
        torchnlp.utils.set_random_seed(seed)

    train_set = WikisqlDataset(data_path, "train")
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=batchifier)

    tokenizer = AutoTokenizer.from_pretrained(config.get("pretrained_model_name", "bert-base-uncased"))
    vocab = Vocab.from_pretrained_tokenizer(
        tokenizer, pad_token="[PAD]", unk_token="[UNK]", sep_token="[SEP]", cls_token="[CLS]"
    )
    runner = Engine(config, tokenizer, vocab)

    # check training data
    # for batch_data in tqdm(train_loader, desc="Checking Training Data"):
    #     runner.prepare_batch(batch_data)

    train_tqdm = tqdm(train_loader)
    try:
        for batch_data in train_tqdm:
            loss = runner.feed(batch_data)
            train_tqdm.set_description(f"loss: {loss}")
    except Exception as e:
        runner.save_checkpoint(save_path)
