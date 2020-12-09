from logging import getLogger

from sklearn.metrics import classification_report

from dajare_detector.utils.base_task import DajareTask
from dajare_detector.predict.predict_lgbm import PredictLGBM
from dajare_detector.make_data.make_multiple_feature_dataset import MakeMultipleFeatureDataset
from dajare_detector.make_data.make_sampling_dataset import MakeSamplingDataset
from dajare_detector.make_data.make_dataset import MakeDataset
from dajare_detector.featurize.add_text_tfidf_feature import AddTextTfidfFeature
from dajare_detector.featurize.add_co_occurrence_roman_feature import AddCoOOccurrenceRomanFeature
from dajare_detector.featurize.add_co_occurrence_kana_feature import AddCoOOccurrenceKanaFeature

logger = getLogger(__name__)


class EvalMultipleFeatureSampling(DajareTask):
    """複数の特徴量のデータで評価"""
    def requires(self):
        target = MakeSamplingDataset(target=MakeDataset())
        feature = MakeMultipleFeatureDataset(target=target)
        train_test_val = AddTextTfidfFeature(target=feature)
        for i in range(2, 5):
            train_test_val = AddCoOOccurrenceRomanFeature(
                train_test_val=train_test_val, target=target, min_roman=i)
        for i in range(2, 5):
            train_test_val = AddCoOOccurrenceKanaFeature(
                train_test_val=train_test_val,
                target=target,
                split_window_size=i)
        return PredictLGBM(target=feature, train_test_val=train_test_val)

    def run(self):
        df = self.load_data_frame()
        report = classification_report(
            df['dajare'], df['pred'].apply(lambda x: 1 if x >= 0.5 else 0))
        logger.info(f'classification_report:{report}')
        self.dump(report)
