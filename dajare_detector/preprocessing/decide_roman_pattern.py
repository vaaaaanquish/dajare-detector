from logging import getLogger

import gokart
import luigi
from fuzzysearch import find_near_matches

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class DecideRomanPattern(DajareTask):
    """ローマ字パターンに繰り返しがあるか判定"""
    target = gokart.TaskInstanceParameter()
    min_roman = luigi.IntParameter()

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        df['decide_roman_flag_list'] = df['roman_pattern'].apply(
            self._decide_roman_pattern)
        self.dump(df[['_id', 'decide_roman_flag_list']])

    def _decide_roman_pattern(self, roman_pattern):
        flag_list = []
        for x in roman_pattern:
            if self._decide_roman(x):
                flag_list.append(True)
            else:
                flag_list.append(False)
        return flag_list

    def _decide_roman(self, sentence):
        words = {x for x in sentence.split(' ') if len(x) > self.min_roman}
        choices = ''.join(sentence.split(' '))
        for word in words:
            if len([
                    x for x in find_near_matches(word, choices, max_l_dist=1)
                    if len(x.matched) >= len(word) and x.matched != word
            ]) > 1:
                return True
        return False
