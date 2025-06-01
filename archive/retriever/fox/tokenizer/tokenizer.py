from abc import ABC, abstractmethod
from typing import Union, Dict, List, Any, Optional

class BaseTokenizer(ABC):
    
    def __call__(self, text: Union[List, str], *args, **kwargs):
        return self.tokenize(text, *args, **kwargs)
    
    @abstractmethod
    def list_saved(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError("`list_saved()` is not implemented properly!")
    
    @abstractmethod
    def from_file(self, dirname:str, *args, **kwargs) -> None:
        raise NotImplementedError("`from_file()` is not implemented properly!")
    
    @abstractmethod
    def expand_vocab(self, vocabs: List[str], *args, **kwargs) -> None:
        raise NotImplementedError("`expand_vocab()` is not implemented properly!")
    
    @abstractmethod
    def tokenize(
        self,
        text: Union[List, str],
        lemmatized: bool=False,
        *args, **kwargs
    ) -> Union[Dict, List]:
        raise NotImplementedError("`tokenize()` is not implemented properly!")
    
    @abstractmethod
    def lemmatize(
        self,
        sentence: str,
        neversplit: Optional[List]=None,
        *args, **kwargs
    ) -> List:
        raise NotImplementedError("`lemmatize()` is not implemented properly!")
    
    @abstractmethod
    def save(self, name:str, overwrite:bool) -> None:
        raise NotImplementedError("`save()` is not implemented properly!")