from logging import getLogger

import gokart
import luigi
import pandas as pd

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.model.train_co_occurrence_roman import TrainCoOccurrenceRoman
from dajare_detector.preprocessing.decide_roman_pattern import DecideRomanPattern
from dajare_detector.preprocessing.make_roman_pattern import MakeRomanPattern
from dajare_detector.preprocessing.make_kana_pattern import MakeKanaPattern
from dajare_detector.preprocessing.normalize_kana_pattern import NormalizeKanaPattern

logger = getLogger(__name__)


class AddCoOOccurrenceRomanFeature(DajareTask):
    """trainデータから音の共起出してfeatureに追加"""
    target = gokart.TaskInstanceParameter()
    train_test_val = gokart.TaskInstanceParameter()
    min_roman = luigi.IntParameter()

    def requires(self):
        roman = MakeRomanPattern(target=NormalizeKanaPattern(
            target=MakeKanaPattern(target=self.target)))
        decided = DecideRomanPattern(target=roman, min_roman=self.min_roman)
        return {
            'model':
            TrainCoOccurrenceRoman(decide_roman_data=decided,
                                   train_test_val=self.train_test_val),
            'train_test_val':
            self.train_test_val,
            'decided':
            decided
        }

    def run(self):
        decided = self.load_data_frame('decided').reset_index(drop=True)
        data = self.load('train_test_val')
        vectorizer = self.load('model')

        # vectorを追加
        for x in ['train_features', 'test_features', 'valid_features']:
            df = pd.merge(data[x], decided, on='_id', how='left')
            df['feature'] = df.apply(lambda x: self._vectorize(x, vectorizer),
                                     axis=1)
            data[x] = df[['_id', 'feature']]

        self.dump(data)

    def _vectorize(self, row, vectorizer):
        return row['feature'] + vectorizer.transform([self._select_word(row)])

    def _select_word(self, row):
        for word_set in row['decide_roman_word_list']:
            if len(word_set) > 0:
                return sorted(word_set)
        return ['', '']
