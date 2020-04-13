""" data """
import logging
import re
from collections import Counter
from operator import itemgetter
from typing import Collection, Dict, Iterable, Iterator, List, Mapping, Union

from transformers import PreTrainedTokenizer

from torchnlp.engine import PytorchStateMixin

logger = logging.getLogger(__name__)

MARK = object()
ATTR_TOKEN_REGEX = re.compile(r"(?P<prefix>\w+)_token$")
ATTR_INDEX_REGEX = re.compile(r"(?P<prefix>\w+)_index$")


# Since attribute tokens is passed by kwargs, their order is fixed after python3.6.
class Vocab(PytorchStateMixin, Collection[str]):
    """
    :attr _index2token
    """

    def __init__(self, token2index: Mapping[str, int] = None, **attribute_tokens: str):
        if isinstance(token2index, Counter):
            raise ValueError("you should not pass counter to Vocab's constructor")
        self._token2index = token2index or {}
        self._index2token = {index: token for token, index in self._token2index.items()}
        self.__setup_attribute_tokens(attribute_tokens)

    def __setup_attribute_tokens(self, attribute_tokens: Dict[str, str]):
        """ expose """
        # expose attribute_tokens
        self.__attribute_tokens = attribute_tokens
        missing_tokens = []
        for token_attr, token in attribute_tokens.items():
            # only token that end with '_token' will be exposed to vocabulary's attribute
            if not token_attr.endswith("_token"):
                continue
            index = self._token2index.get(token, MARK)
            if index is MARK:
                missing_tokens.append((token_attr, token))
                continue
            index_attr = ATTR_TOKEN_REGEX.sub(r"\g<prefix>_index", token_attr)
            setattr(self, token_attr, token)
            setattr(self, index_attr, index)

        if missing_tokens:
            logger.warning(missing_tokens)

    # def __remove_attribute_tokens(self):
    #     for attr_token in self.__attribute_tokens:
    #         delattr(self, attr_token)
    def state_dict(self):
        """ state dict """
        return {"token2index": self._token2index, "attribute_tokens": self.__attribute_tokens}

    def load_state_dict(self, state_dict):
        """ load state """
        attribute_tokens = state_dict.pop("attribute_tokens")
        token2index = state_dict.get("token2index")
        self._token2index = token2index
        self._index2token = {index: token for token, index in self._token2index.items()}
        self.__setup_attribute_tokens(attribute_tokens)
        return self

    def __getattr__(self, attr):  # could not be find
        msg = "{attr} is not set, but accessed, you add {added}, forgot pass it to constructor?".format(
            attr=attr, added=self.__attribute_tokens
        )
        raise ValueError(msg)

    @classmethod
    def from_counter(cls, counter: Mapping[str, int], max_size: int = None, **attr_tokens: str):
        """ from counter """
        counter = Counter(counter)
        size = max_size - len(attr_tokens) if max_size else None
        token2index = {token: index for index, token in enumerate(map(itemgetter(0), counter.most_common(size)))}
        return cls(token2index, **attr_tokens)

    @classmethod
    def from_iterable(cls, iterable: Iterable[str], **attr_tokens):
        """ from iterable """
        token2index = {token: index for index, token in enumerate(set(iterable))}
        return cls(token2index, **attr_tokens)

    @classmethod
    def from_pretrained_tokenizer(cls, tokenizer: PreTrainedTokenizer, **attribute_tokens: str):
        """ retrieve tokenizer from pretrained_tokenizer """
        tokenizer_tokens = {}
        for attr_name in tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            token = getattr(tokenizer, attr_name)
            if token is not None:
                tokenizer_tokens[attr_name] = token

        attribute_tokens.update(tokenizer_tokens)
        return cls(tokenizer.get_vocab(), **attribute_tokens)

    def item2index(self, token: str) -> int:
        """ convert token to index """
        index = self._token2index.get(token, None)
        if index is None:
            return self.unk_index
        return index

    def index2item(self, index: int) -> str:
        """ convert index to token """
        return self._index2token[index]

    def map2index(self, tokens: Iterable[str]) -> List[int]:
        """ map """
        return [self.item2index(token) for token in tokens]

    def map2item(self, indices: Iterable[int]) -> List[str]:
        """ unmap """
        return [self.index2item(index) for index in indices]

    def wild_tokens2indices(self, nested_tokens: List[Union[List, str]]):
        """ wild conversion """
        return [self.item2index(s) if isinstance(s, str) else self.wild_tokens2indices(s) for s in nested_tokens]

    def __len__(self):
        return len(self._index2token)

    def __iter__(self) -> Iterator[str]:
        return iter(self._token2index)

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and token in self._token2index
