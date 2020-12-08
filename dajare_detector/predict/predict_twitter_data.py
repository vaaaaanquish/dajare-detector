from logging import getLogger

import pandas as pd
import luigi
import swifter  # noqa
import numpy as np

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.featurize.add_text_tfidf_feature import AddTextTfidfFeature
from dajare_detector.preprocessing.tokenize_data import TokenizeData
from dajare_detector.make_data.make_dataset import MakeDataset
from dajare_detector.make_data.make_twitter_dataset import MakeTwitterDataset
from dajare_detector.make_data.make_multiple_feature_dataset import MakeMultipleFeatureDataset
from dajare_detector.model.train_lgbm import TrainLGBM
from dajare_detector.model.train_tfidf import TrainTfidf

logger = getLogger(__name__)


class PredictTwitterData(DajareTask):
    file_path = luigi.Parameter()

    def requires(self):
        feature = MakeMultipleFeatureDataset(target=MakeDataset())
        tokenized = TokenizeData(target=MakeDataset())
        train_test_val = AddTextTfidfFeature(target=feature)
        tfidf = TrainTfidf(tokenized_data=tokenized,
                           train_test_val=train_test_val)

        tw_data = MakeTwitterDataset(file_path=self.file_path)
        tw_feature = MakeMultipleFeatureDataset(target=tw_data)
        tw_tokenized = TokenizeData(target=tw_data)

        return {
            'model': TrainLGBM(target=train_test_val),
            'tfidf': tfidf,
            'feature': tw_feature,
            'tokenized': tw_tokenized
        }

    def run(self):
        model = self.load('model')
        vectorizer = self.load('tfidf')
        tweet_feature = self.load_data_frame('feature').reset_index(drop=True)
        tweet_tokenized = self.load_data_frame('tokenized').reset_index(
            drop=True)

        col = [
            x for x in tweet_feature.columns
            if x not in {'_id', 'text', 'dajare', 'rule_base'}
        ]
        tweet_feature = pd.merge(tweet_feature,
                                 tweet_tokenized[['_id', 'tokenized_text']],
                                 on='_id')
        tweet_feature['feature'] = tweet_feature.swifter.apply(
            lambda x: self._explode_cols(x, col), axis=1).tolist()
        tweet_feature['feature'] = tweet_feature.apply(
            lambda x: self._vectorize(x, vectorizer), axis=1)
        pred = model.predict(tweet_feature['feature'].tolist(),
                             num_iteration=model.best_iteration)

        if 'rule_base' in tweet_feature.columns:
            self.dump(
                pd.DataFrame({
                    'pred': pred,
                    '_id': tweet_feature['_id'].tolist(),
                    'rule_base': tweet_feature['rule_base'].tolist()
                }))
        else:
            self.dump(
                pd.DataFrame({
                    'pred': pred,
                    '_id': tweet_feature['_id'].tolist()
                }))

    def _explode_cols(self, row, col):
        f = []
        for x in col:
            if type(row[x]) in [int, float]:
                f.append(row[x])
            elif type(row[x]) in [bool]:
                f.append(int(row[x]))
            elif type(row[x]) == np.ndarray:
                f += row[x].tolist()
            elif type(row[x]) == list:
                f += row[x]
        return f

    def _vectorize(self, row, vectorizer):
        return row['feature'] + vectorizer.transform([row['tokenized_text']
                                                      ]).toarray()[0].tolist()
