from logging import getLogger

import gokart
import pandas as pd

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.featurize.make_decide_kana_feature import MakeDecideKanaFeature
from dajare_detector.featurize.make_decide_roman_feature import MakeDecideRomanFeature

logger = getLogger(__name__)


class MakeSamplingDataset(DajareTask):
    """データセットBを生成
    - 青空文庫(負例)
        - ルールベースで誤検出したもの(fp)
        - 謝検出されなかったデータからサンプリング(tn)
    - ダジャレデータセット(正例)
        - ルールベースで検出できなかったもの(fn)
        - 検出できたものからサンプリング(tp)
    """
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

        tasks = {'text': self.target}
        tasks.update(kana_tasks)
        tasks.update(roman_tasks)
        return tasks

    def run(self):
        df = self.load_data_frame('text').reset_index(drop=True)
        kana = [f'kana{i}' for i in range(2, 5)]
        roman = [f'roman{i}' for i in range(2, 5)]
        for x in roman + kana:
            vector_df = self.load_data_frame(x).reset_index(drop=True)
            df = pd.merge(df, vector_df, on='_id').reset_index(drop=True)
        col = [x for x in df.columns if x.startswith('decide_')]
        df['decide_rule'] = df.apply(lambda x: self._decide_rule(x, col),
                                     axis=1)
        df = self._sampling(df)
        self.dump(df[['_id', 'text', 'dajare', 'decide_rule']])

    def _decide_rule(self, row, col):
        """ルールベースでどこかに引っかかればTrue"""
        for x in col:
            if row[x] > 0:
                return True
        return False

    def _sampling(self, df):
        """正負サンプリング. 数が足りない場合は小さい方に合わせる"""
        fp = df[(df['dajare'] == 0) & df['decide_rule']]
        tn = df[(df['dajare'] == 0) & ~df['decide_rule']]
        tp = df[(df['dajare'] == 1) & df['decide_rule']]
        fn = df[(df['dajare'] == 1) & ~df['decide_rule']]
        sample_df = pd.DataFrame()
        sample_df = sample_df.append(fp)
        sample_size = len(fp) if len(fp) <= len(tn) else len(tn)
        sample_df = sample_df.append(tn.sample(sample_size))
        sample_df = sample_df.append(fn)
        sample_size = len(fn) if len(fn) <= len(tp) else len(tp)
        sample_df = sample_df.append(tp.sample(sample_size))
        return sample_df
