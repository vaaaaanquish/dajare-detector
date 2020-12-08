from logging import getLogger
import gc

import gokart
from sklearn.feature_extraction.text import TfidfVectorizer

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class TrainTfidf(DajareTask):
    """train dataの単語のリストからTF-IDF """
    tokenized_data = gokart.TaskInstanceParameter()
    train_test_val = gokart.TaskInstanceParameter()

    def requires(self):
        return {
            'tokenized_data': self.tokenized_data,
            'train_test_val': self.train_test_val
        }

    def run(self):
        df = self.load_data_frame('tokenized_data').reset_index(drop=True)
        train_test_val = self.load('train_test_val')

        df = df[df['_id'].isin(train_test_val['train_features']['_id'])]
        del train_test_val
        gc.collect()

        vectorizer = TfidfVectorizer(max_features=1000)
        vectorizer.fit(df['tokenized_text'].tolist())
        self.dump(vectorizer)
