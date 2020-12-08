from logging import getLogger

import gokart
import pandas as pd

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.featurize.make_bert_feature import MakeBertFeature
from dajare_detector.featurize.make_decide_kana_feature import MakeDecideKanaFeature
from dajare_detector.featurize.make_decide_roman_feature import MakeDecideRomanFeature
from dajare_detector.featurize.make_sentence_length_feature import MakeSentenceLengthFeature
from dajare_detector.featurize.make_word_count_feature import MakeWordCountFeature

logger = getLogger(__name__)


class MakeMultipleFeatureDataset(DajareTask):
    """複数の特徴量でデータセット生成(train test分けないもの)"""
    target = gokart.TaskInstanceParameter()

    def requires(self):
        kana_tasks = {
            f'kana{i}': MakeDecideKanaFeature(target=self.target,
                                              split_window_size=i)
            for i in range(2, 5)
        }
        roman_tasks = {
            f'roman{i}': MakeDecideRomanFeature(target=self.target,
                                                min_roman=i)
            for i in range(2, 5)
        }

        tasks = {
            'text': self.target,
            'bert': MakeBertFeature(target=self.target),
            'length': MakeSentenceLengthFeature(target=self.target),
            'count': MakeWordCountFeature(target=self.target),
        }
        tasks.update(kana_tasks)
        tasks.update(roman_tasks)
        return tasks

    def run(self):
        df = self.load_data_frame('text').reset_index(drop=True)
        kana = [f'kana{i}' for i in range(2, 5)]
        roman = [f'roman{i}' for i in range(2, 5)]
        for x in ['bert', 'length', 'count'] + roman + kana:
            vector_df = self.load_data_frame(x).reset_index(drop=True)
            df = pd.merge(df, vector_df, on='_id').reset_index(drop=True)
        self.dump(df)
