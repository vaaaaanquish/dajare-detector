from logging import getLogger

import gokart
import swifter  # noqa
import regex

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.utils.mecab_tokenizer import make_mecab_tagger

logger = getLogger(__name__)


class TokenizeData(DajareTask):
    """mecabによる形態素解析を行い名詞、形容詞、動詞のみ抽出."""
    target = gokart.TaskInstanceParameter()
    kanji_regex = regex.compile(r'\p{Script_Extensions=Han}+')

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        mecab = make_mecab_tagger(self.workspace_directory)
        df['tokenized_text'] = df['text'].swifter.apply(
            lambda x: self._tokenize(x, mecab))
        self.dump(df[['_id', 'tokenized_text']])

    def _tokenize(self, text, mecab):
        """
        mecab parse example:
            布団    フトン  布団    名詞-一般
            が      ガ      が      助詞-格助詞-一般
            ふっとん        フットン        ふっとぶ        動詞-自立       五段・バ行      連用タ接続
            だ      ダ      だ      助動詞  特殊・タ        基本形
            EOS
        """
        words = []
        for x in mecab.parse(text).splitlines():
            y = x.split()
            if len(y) < 3:
                continue
            if y[3].startswith('名詞') or y[3].startswith(
                    '形容詞') or y[3].startswith('動詞'):
                if len(y[0]) > 1 or self.kanji_regex.fullmatch(y[0]):
                    words.append(y[0])
        return ' '.join(words)
