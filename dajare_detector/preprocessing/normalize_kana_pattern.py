from logging import getLogger
import re

import alkana
import gokart

from dajare_detector.utils.base_task import DajareTask

logger = getLogger(__name__)


class NormalizeKanaPattern(DajareTask):
    """カタカナをノーマライズ"""
    alpha_regex = re.compile(r'[a-zA-Z]+')
    target = gokart.TaskInstanceParameter()
    replace_words = [['。', ' '], ['、', ' '], [', ', ' '], ['.',
                                                           ' '], ['!', ' '],
                     ['！', ' '], ['・', ' '], ['「', ' '], ['」',
                                                          ' '], ['「', ' '],
                     ['｣', ' '], ['『', ' '], ['』', ' '], ['　',
                                                          ' '], ['ッ', ' '],
                     ['ャ', 'ヤ'], ['ュ', 'ユ'], ['ョ', 'ヨ'], ['ァ',
                                                          'ア'], ['ィ', 'イ'],
                     ['ゥ', 'ウ'], ['ェ', 'エ'], ['ォ', 'オ'], ['ー', ' '],
                     ['"', ' '], ["'", ' ']]

    def requires(self):
        return self.target

    def run(self):
        df = self.load_data_frame().reset_index(drop=True)
        df['normalized_kana_pattern'] = df['kana_pattern'].apply(
            self._normalize)
        self.dump(df[['_id', 'normalized_kana_pattern']])

    def _normalize(self, sentences):
        normalized_pattern = []
        for sentence in sentences:
            for alpha in self.alpha_regex.findall(sentence):
                kana = alkana.get_kana(alpha.lower())
                sentence = sentence.replace(alpha, kana) if kana else sentence
            for i, replace_word in enumerate(self.replace_words):
                sentence = sentence.replace(replace_word[0], replace_word[1])
            if sentence not in normalized_pattern:
                normalized_pattern.append(sentence)
        return normalized_pattern
