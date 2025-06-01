import logging
from typing import List
from fox.utils.startup import config
try:
    from MeCab import Tagger
except ImportError as e:
    logging.error("MeCab is not installed. Install the MeCab first.")
    
class KoMecabTokenizer():
    # `mecab-ko-dic` path: default path with mecab installation
    # To add custom user-dictionary to the Mecab tokenizer, please refer to the README as below:
    # `~/mecab/mecab-ko-dic-2.1.1-20180720/user-dic/README.md`
    
    def __init__(self):       
        self.mecab = Tagger(f'-d {config.get("mecab_dic_path")}')
        self.max_seq_len = 10000 # mecab acceptable sequence length
    
    @staticmethod
    def _parse_morpheme(morpheme: str):
        if not morpheme:
            return ('', 'SY')
        morpheme_splits = morpheme.split('\t', 1)
        if len(morpheme_splits) != 2:
            return ('', 'SY')
        surface, tag = morpheme_splits
        tag = tag.split(',', 1)[0]
        return (surface, tag)
    
    def _split_sent(self, sent: str) -> List[str]:
        eojeols = sent.split(" ")
        split_sents = [' '.join(eojeols[i:i+self.max_seq_len]) for i in range(0, len(eojeols), self.max_seq_len)]
        return split_sents
    
    def pos(self, sent: str) -> List[tuple]:
        """Parse position of speech given sentence.
        
        Args:
            sent (str): sentence to analyze
        """
        # When len(sent) exceeds the max_seq_len, the MeCab Tagger return `None` instead of the pos.
        # To prevent it, here to split sentence by eojeols and rerun the parse function.
        eojeols = sent.split(" ")
        if len(eojeols) > self.max_seq_len:
            splits = []
            split_sents = self._split_sent(sent)
            for _sent in split_sents:
                _result = self.mecab.parse(_sent)
                if _result is not None:
                    splits.append(_result.replace('EOS\n', ''))
            splits.append('EOS\n')
            result = ''.join(splits)
        else:
            result = self.mecab.parse(sent)
        if result is None:
            raise ValueError(f"The sentence is not tokenizable with Mecab. | len(sent): {len(eojeols)} | sent: {sent}")
        pos = [self._parse_morpheme(morpheme) for morpheme in result.splitlines()]
        pos = list(filter(lambda x: x[0] != "", pos))
        return pos
        
    def tokenize(self, sent: str) -> List[str]:
        """Tokenize sentence.
        
        Args:
            sent (str): sentence to tokenize
        """
        pos = self.pos(sent)
        morphs = [surface for surface, _ in pos]
        return morphs
            
    