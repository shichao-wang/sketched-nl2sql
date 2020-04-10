""" config container """
from numbers import Number
from typing import Dict, Mapping, Union

ConfigValueType = Union[Number, str]


class Config(Dict[str, ConfigValueType]):
    """ config """

    def __init__(self, mapping: Mapping[str, ConfigValueType] = None, **kwargs):
        super().__init__(**mapping or {}, **kwargs)


# class HierarchicalConfig(Dict[str, Union[ConfigValueType, "HierarchicalConfig"]]):
#     pass
