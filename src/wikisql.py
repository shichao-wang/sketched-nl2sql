# """ wikisql """
# import operator
# from typing import NamedTuple, Set, List
#
#
# class Cond:
#     """ raw wikisql query """
#
#     col_id: int
#     op_id: int
#     value: str
#
#
# class Query(NamedTuple):
#     """ raw wikisql query """
#
#     col_id: int
#     agg_id: int
#     conds: Set[Cond] = set()
#
#
# class Example(NamedTuple):
#     """ data class represent raw wikisql example """
#
#     phase: int
#     question: str
#     table_id: int
#     query: Query
#
#
# class Table(NamedTuple):
#     """ wikisql table """
#
#     header: List[str]
#     types: List[str]
#
#
# class TableRepository:
#     def __init__(self, table_file: str):
#         pass
#
#
# def read_data(data_file: str, table_file: str):
#     """
#     :param data_file:
#     :param table_file:
#     :return:
#     """
#     dict(zip(operator.itemgetter("id"),))
#     with open(table_file) as fp:
#         pass
