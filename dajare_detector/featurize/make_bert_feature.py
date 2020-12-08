from logging import getLogger

import gokart
import swifter  # noqa

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.utils.bert_vectorizer import load_sentence_bert

logger = getLogger(__name__)


class MakeBertFeature(DajareTask):
    """bertによる特徴量化"""
    target = gokart.TaskInstanceParameter()

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        model = load_sentence_bert(self.workspace_directory)
        logger.info('vectorize...')
        df['bert_vector'] = df['text'].swifter.apply(
            lambda x: self._vectorize(x, model))
        self.dump(df[['_id', 'bert_vector']])

    def _vectorize(self, text, model):
        return model.encode([text])[0]
