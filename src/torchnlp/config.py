""" config container """

from typing import Dict, Mapping


class Config(Dict):
    """ config """

    def __init__(self, mapping: Mapping = None, **kwargs):
        mapping = mapping or {}
        super().__init__(**mapping, **kwargs)

    def get_int(self, item) -> int:
        value = self.get(item)
        assert isinstance(value, int)
        return int(value)

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """ load config from yaml file """
        import yaml

        with open(yaml_file, encoding="utf-8") as fp:
            content = yaml.load(fp, yaml.FullLoader)
        return cls(content)
