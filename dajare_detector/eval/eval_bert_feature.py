from logging import getLogger

from sklearn.metrics import classification_report

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.predict.predict_lgbm import PredictLGBM
from dajare_detector.make_data.make_bert_feature_dataset import MakeBertFeatureDataset
from dajare_detector.make_data.make_dataset import MakeDataset
from dajare_detector.preprocessing.train_test_val_split import TrainTestValSplit

logger = getLogger(__name__)


class EvalBertFeature(DajareTask):
    """sentence bertのみで評価"""
    def requires(self):
        feature = MakeBertFeatureDataset(target=MakeDataset())
        return PredictLGBM(target=feature,
                           train_test_val=TrainTestValSplit(target=feature))

    def run(self):
        df = self.load_data_frame()
        report = classification_report(
            df['dajare'], df['pred'].apply(lambda x: 1 if x >= 0.5 else 0))
        logger.info(f'classification_report:{report}')
        self.dump(report)
