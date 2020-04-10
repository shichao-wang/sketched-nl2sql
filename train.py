import logging

import sketched_nl2sql
from torchnlp.config import Config

DATA_PATH = "/Users/chaoftic/Public/数据集/WikiSQL"
config = Config(pretrained_model_name="/Users/chaoftic/Public/bert-base-uncased", hidden_dim=300, batch_size=32)

logging.basicConfig(filename="wikisql.log", level=logging.DEBUG)
sketched_nl2sql.train(DATA_PATH, "save", config)
