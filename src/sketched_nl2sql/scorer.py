import logging
import sqlite3

from sketched_nl2sql.dataset import Example

logger = logging.getLogger(__name__)


def exact_accuracy(pred_example: Example, gold_example: Example):
    """ exact match """
    return pred_example.query == gold_example.query


def execution_accuracy(pred_query_string: str, gold_query_string: str, conn: sqlite3.Connection):
    """ logical match """
    try:
        pred = conn.execute(pred_query_string).fetchall()
        gold = conn.execute(gold_query_string).fetchall()
        return pred == gold
    except sqlite3.Error:
        return False
