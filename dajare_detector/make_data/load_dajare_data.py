from logging import getLogger

import luigi
import pandas as pd

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class LoadDajareData(DajareTask):
    file_path = luigi.Parameter()

    def run(self):
        df = pd.read_csv(self.file_path)
        df['_id'] = pd.Series(list(range(len(df))))
        self.dump(df)
