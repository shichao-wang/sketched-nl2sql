""" config container """
from numbers import Number
from typing import Dict, Mapping, Union

ConfigValueType = Union[Number, str]


class Config(Dict[str, ConfigValueType]):
    """ config """

    def __init__(self, mapping: Mapping[str, ConfigValueType] = None, **kwargs):
        super().__init__(**mapping or {}, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """ load config from yaml file """
        import yaml

        with open(yaml_file, encoding="utf-8") as fp:
            content = yaml.load(fp, yaml.FullLoader)
        return cls(content)


# class HierarchicalConfig(Dict[str, Union[ConfigValueType, "HierarchicalConfig"]]):
#     pass
