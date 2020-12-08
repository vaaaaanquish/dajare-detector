from logging import getLogger

import luigi
import swifter  # noqa
import gokart

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.utils.mecab_tokenizer import make_mecab_tagger

logger = getLogger(__name__)


class MakeKanaPattern(DajareTask):
    """mecabによって複数のカタカナ読みのパターンを生成."""
    mecab_kana_pattern_num = luigi.IntParameter()
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        mecab = make_mecab_tagger(self.workspace_directory)
        df['kana_pattern'] = df['text'].swifter.apply(
            lambda x: self._make_kana_pattern(x, mecab))
        self.dump(df[['_id', 'kana_pattern']])

    def _make_kana_pattern(self, text, mecab):
        """順番を保持しつつset"""
        mecab.parseNBestInit(text)
        pattern = []
        for _ in range(self.mecab_kana_pattern_num):
            x = self._make_kana(mecab)
            if x not in pattern:
                pattern.append(x)
        return pattern

    def _make_kana(self, mecab):
        node = mecab.nextNode()
        kana = []
        while node:
            n = node.feature.split(',')
            if len(n) == 9:
                kana.append(n[7].replace('*', ''))
            node = node.next
        return ' '.join(kana).strip()
