from logging import getLogger

import gokart

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.utils.co_occurrence_vectorizer import CoOccurrenceVectorizer

logger = getLogger(__name__)


class TrainCoOccurrenceRoman(DajareTask):
    """train dataの重複音のリストから共起頻度を取る"""
    decide_roman_data = gokart.TaskInstanceParameter()
    train_test_val = gokart.TaskInstanceParameter()

    def requires(self):
        return {
            'decide_roman_data': self.decide_roman_data,
            'train_test_val': self.train_test_val
        }

    def run(self):
        df = self.load_data_frame('decide_roman_data').reset_index(drop=True)
        df = df[df['_id'].isin(
            self.load('train_test_val')['train_features']['_id'])]
        df['word_set'] = df.apply(self._select_word, axis=1)
        df = df[df['word_set'].apply(lambda x: len(x) == 2)]
        vectorizer = CoOccurrenceVectorizer()
        vectorizer.fit(df['word_set'].tolist())
        self.dump(vectorizer)

    def _select_word(self, row):
        """一番最初の重複音を返す"""
        for word_set in row['decide_roman_word_list']:
            if len(word_set) > 0:
                return sorted(word_set)
        return []
