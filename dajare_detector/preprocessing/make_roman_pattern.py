from logging import getLogger

import gokart
from pykakasi import kakasi

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class MakeRomanPattern(DajareTask):
    """ローマ字パターンを生成"""
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        hepburn, kunrei, passport = self._make_kakasi()
        df['roman_pattern'] = df['normalized_kana_pattern'].apply(
            lambda x: self._make_roman_pattern(x, hepburn, kunrei, passport))
        self.dump(df[['_id', 'roman_pattern']])

    def _make_kakasi(self):
        conv_list = []
        for r in ['Hepburn', 'Kunrei', 'Passport']:
            kks = kakasi()
            kks.setMode('K', 'a')
            kks.setMode('r', r)
            conv_list.append(kks.getConverter())
        return conv_list

    def _make_roman_pattern(self, kana_pattern, hepburn, kunrei, passport):
        roman_pattern = set()
        for k in kana_pattern:
            roman_pattern.add(hepburn.do(k))
            roman_pattern.add(kunrei.do(k))
            roman_pattern.add(passport.do(k))
        return list(roman_pattern)
