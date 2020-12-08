from logging import getLogger
from collections import Counter

import gokart
import pandas as pd
import luigi

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class DecideKanaPattern(DajareTask):
    """カナの繰り返し判定"""
    split_pattern_target = gokart.TaskInstanceParameter()
    kana_pattern_target = gokart.TaskInstanceParameter()
    split_window_size = luigi.IntParameter()

    def requires(self):
        return {
            'split': self.split_pattern_target,
            'kana': self.kana_pattern_target
        }

    def run(self):
        split_df = self.load_data_frame('split').reset_index(drop=True)
        kana_df = self.load_data_frame('kana').reset_index(drop=True)
        df = pd.merge(split_df, kana_df, on='_id')
        df['decide_kana_flag_list'] = df[[
            'splited_pattern', 'normalized_kana_pattern'
        ]].apply(self._decide_kana_pattern, axis=1)
        self.dump(df[['_id', 'decide_kana_flag_list']])

    def _decide_kana_pattern(self, row):
        flag_list = []
        for kana, splited_kana in zip(row['normalized_kana_pattern'],
                                      row['splited_pattern']):
            if len(splited_kana) == 0:
                flag_list.append(False)
                continue
            word, count = self._get_top_of_count_word(splited_kana)
            if count > 1 and self._sentence_max_dup_rate(
                    word, kana, splited_kana) <= 0.5:
                flag_list.append(True)
            else:
                flag_list.append(False)
        return flag_list

    def _get_top_of_count_word(self, splited_kana):
        return Counter(splited_kana).most_common()[0]

    def _sentence_max_dup_rate(self, sentence, preprocessed, splited):
        if self.split_window_size == 2:
            return 0.0 if preprocessed.replace(sentence, '') else 1.0
        return splited.count(sentence) / len(set(splited))
