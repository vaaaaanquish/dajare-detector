from collections import Counter
import re

import MeCab
import alkana
from pykakasi import kakasi
from fuzzysearch import find_near_matches


class DajareDetector:
    """Refactored Shareka."""
    def __init__(self,
                 neologd,
                 window_size=2,
                 min_roman=2,
                 mecab_pattern_num=50):
        self.window_size = window_size
        self.mecab_pattern_num = mecab_pattern_num
        self.min_roman = min_roman
        self.replace_words = [['。', ''], ['、', ''], [', ', ''], ['.', ''],
                              ['!', ''], ['！', ''], ['・', ''], ['「', ''],
                              ['」', ''], ['「', ''], ['｣', ''], ['『', ''],
                              ['』', ''], [' ', ''], ['　', ''], ['ッ', ''],
                              ['ャ', 'ヤ'], ['ュ', 'ユ'], ['ョ', 'ヨ'], ['ァ', 'ア'],
                              ['ィ', 'イ'], ['ゥ', 'ウ'], ['ェ', 'エ'], ['ォ', 'オ'],
                              ['ー', ''], ['"', ''], ["'", '']]
        self.alpha_regex = re.compile(r'[a-zA-Z]+')
        self.mecab = MeCab.Tagger(f'-Ochasen -d {neologd}')
        self.hepburn, self.kunrei, self.passport = self._make_kakasi()

    def do(self, sentence):
        self.mecab.parseNBestInit(sentence)
        kana_pattern = {
            self._make_kana_pattern()
            for _ in range(self.mecab_pattern_num)
        }
        roman_pattern = self._make_roman_pattern(kana_pattern)
        normalized_kana_pattern = {self._normalize(x) for x in kana_pattern}
        devided_pattern = [self._devide(x) for x in normalized_kana_pattern]
        if self._decide_kana_pattern(normalized_kana_pattern, devided_pattern):
            return True
        elif self._decide_roman_pattern(roman_pattern):
            return True
        return False

    def _make_kakasi(self):
        conv_list = []
        for r in ['Hepburn', 'Kunrei', 'Passport']:
            kks = kakasi()
            kks.setMode('K', 'a')
            kks.setMode('r', r)
            conv_list.append(kks.getConverter())
        return conv_list

    def _make_kana_pattern(self):
        node = self.mecab.nextNode()
        kana = []
        while node:
            n = node.feature.split(',')
            if len(n) == 9:
                kana.append(n[7].replace('*', ''))
            node = node.next
        return ' '.join(kana).strip()

    def _make_roman_pattern(self, kana_pattern):
        roman_pattern = set()
        for k in kana_pattern:
            roman_pattern.add(self.hepburn.do(k))
            roman_pattern.add(self.kunrei.do(k))
            roman_pattern.add(self.passport.do(k))
        return roman_pattern

    def _normalize(self, sentence):
        for alpha in self.alpha_regex.findall(sentence):
            kana = alkana.get_kana(alpha.lower())
            sentence = sentence.replace(alpha, kana) if kana else sentence
        for i, replace_word in enumerate(self.replace_words):
            sentence = sentence.replace(replace_word[0], replace_word[1])
        return sentence

    def _devide(self, sentence):
        repeat_num = len(sentence) - (self.window_size - 1)
        return [sentence[i:i + self.window_size] for i in range(repeat_num)]

    def _decide_kana_pattern(self, normalized_kana_pattern, devided_pattern):
        for kana, devided_kana in zip(normalized_kana_pattern,
                                      devided_pattern):
            if len(devided_kana) == 0:
                continue
            word, count = self._get_top_of_count_word(devided_kana)
            if count > 1 and self._sentence_max_dup_rate(
                    word, kana, devided_kana) <= 0.5:
                return True
        return False

    def _decide_roman_pattern(self, roman_pattern):
        for x in roman_pattern:
            if self._decide_roman(x):
                return True
        return False

    def _get_top_of_count_word(self, devided_kana):
        return Counter(devided_kana).most_common()[0]

    def _sentence_max_dup_rate(self, sentence, preprocessed, devided):
        if self.window_size == 2:
            return 0.0 if preprocessed.replace(sentence, '') else 1.0
        return devided.count(sentence) / len(set(devided))

    def _decide_roman(self, sentence):
        words = {x for x in sentence.split(' ') if len(x) > self.min_roman}
        choices = ''.join(sentence.split(' '))
        for word in words:
            if len([
                    x for x in find_near_matches(word, choices, max_l_dist=1)
                    if len(x.matched) >= len(word) and x.matched != word
            ]) > 1:
                return True
        return False


if __name__ == '__main__':
    DajareDetector().do('これはテストです')
