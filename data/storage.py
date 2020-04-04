from typing import Dict, List, NamedTuple

agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
cond_ops = ["=", ">", "<", "OP"]


class SketchedQuery:
    class Condition(NamedTuple):
        column_index: int
        operator_index: int
        value_start: int
        value_end: int

    def __init__(self, aggregator_index: int, select_index: int, conditions: List[Condition] = None):
        self.aggregate_index = aggregator_index
        self.select_index = select_index
        self.conditions = conditions

    @classmethod
    def from_dict(cls, query_dict: Dict):
        conditions = [
            SketchedQuery.Condition(
                cond["col_id"], cond["op_id"], cond["value_beg"], cond["value_end"]
            )
            for cond in query_dict["conds"]
        ]
        return cls(query_dict["select"], query_dict["aggregator"], conditions)

    def __repr__(self):
        query_string = "SELECT {}({}) FROM table".format(agg_ops[self.aggregate_index], self.select_index)
        if self.conditions:
            query_string += " WHERE "
            query_string += "AND".join(
                [
                    f"{cond.column_index} {cond_ops[cond.operator_index]} [{cond.value_start, cond.value_end}]"
                    for cond in self.conditions
                ]
            )
        return query_string
