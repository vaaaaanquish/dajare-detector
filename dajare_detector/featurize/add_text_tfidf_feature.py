from logging import getLogger

import gokart
import pandas as pd

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.model.train_tfidf import TrainTfidf
from dajare_detector.preprocessing.train_test_val_split import TrainTestValSplit
from dajare_detector.preprocessing.tokenize_data import TokenizeData

logger = getLogger(__name__)


class AddTextTfidfFeature(DajareTask):
    """trainデータから単語のリストからTF-IDFしてfeatureに追加"""
    target = gokart.TaskInstanceParameter()

    def requires(self):
        tokenized = TokenizeData(target=self.target)
        train_test_val = TrainTestValSplit(target=self.target)
        return {
            'model':
            TrainTfidf(tokenized_data=tokenized,
                       train_test_val=train_test_val),
            'train_test_val':
            train_test_val,
            'tokenized':
            tokenized
        }

    def run(self):
        token = self.load_data_frame('tokenized').reset_index(drop=True)
        data = self.load('train_test_val')
        vectorizer = self.load('model')

        # vectorを追加
        for x in ['train_features', 'test_features', 'valid_features']:
            df = pd.merge(data[x], token, on='_id', how='left')
            df['feature'] = df.apply(lambda x: self._vectorize(x, vectorizer),
                                     axis=1)
            data[x] = df[['_id', 'feature']]

        self.dump(data)

    def _vectorize(self, row, vectorizer):
        return row['feature'] + vectorizer.transform([row['tokenized_text']
                                                      ]).toarray()[0].tolist()
