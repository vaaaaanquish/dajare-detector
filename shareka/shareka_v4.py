import MeCab
from collections import Counter
import re
import alkana
from pykakasi import kakasi
from fuzzysearch import find_near_matches


class Shareka:
    alpha_regex = re.compile(r'[a-zA-Z]+')

    def __init__(self, sentence, n=3):
        self.replace_words = [
            ['。', ''], ['、', ''], [', ', ''], ['.', ''], ['!', ''],
            ['！', ''], ['・', ''], ['「', ''], ['」', ''], ['「', ''], ['｣', ''], ['『', ''], ['』', ''], [' ', ''], ['　', ''],
            ['ッ', ''], ['ャ', 'ヤ'], ['ュ', 'ユ'], ['ョ', 'ヨ'],
            ['ァ', 'ア'], ['ィ', 'イ'], ['ゥ', 'ウ'], ['ェ', 'エ'], ['ォ', 'オ'], ['ー', ''], ['"', ''], ["'", '']]
        self.kaburi = n
        self.sentence = sentence
        self.mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        self.kana_pattern = self.make_kana_pattern(sentence)
        self.preprocessed_pattern = self.preprocessing(self.kana_pattern)
        self.devided_pattern = self.devide(self.preprocessed_pattern)
        # romanの音で判定
        self.hepburn, self.kunrei, self.passport = self.make_kakasi()
        self.roman_pattern = self.make_roman_pattern(self.kana_pattern)

    def make_kakasi(self):
        """kakasi"""
        conv_list = []
        for r in ['Hepburn', 'Kunrei', 'Passport']:
            kks = kakasi()
            kks.setMode('K', 'a')
            kks.setMode('r', r)
            conv_list.append(kks.getConverter())
        return conv_list

    def make_roman_pattern(self, kana_pattern):
        """roma字読みに変換"""
        roman_pattern = set()
        for k in kana_pattern:
            roman_pattern.add(self.hepburn.do(k))
            roman_pattern.add(self.kunrei.do(k))
            roman_pattern.add(self.passport.do(k))
        return list(roman_pattern)

    def make_kana_pattern(self, sentence):
        self.mecab.parseNBestInit(sentence)
        kana_pattern = set()
        for _ in range(self.kana_pattern):
            node = self.mecab.nextNode()
            kana = []
            while node:
                n = node.feature.split(",")
                if len(n) == 9:
                    kana.append(n[7].replace('*', ''))
                node = node.next
            kana_pattern.add(' '.join(kana).strip())
        return list(kana_pattern)

    def preprocessing(self, sentence_list):
        result = []
        for sentence in sentence_list:
            # kana変換
            alpha_dict = {x: alkana.get_kana(x.lower()) for x in self.alpha_regex.findall(sentence)}
            for k, v in alpha_dict.items():
                sentence = sentence.replace(k, v) if v else sentence

            for i, replace_word in enumerate(self.replace_words):
                sentence = sentence.replace(replace_word[0], replace_word[1])
            result.append(sentence)
        return result

    def devide(self, sentence_list):
        result = []
        for sentence in sentence_list:
            elements = []
            repeat_num = len(sentence) - (self.kaburi - 1)
            for i in range(repeat_num):
                elements.append(sentence[i:i + self.kaburi])
            result.append(elements)
        return result

    def dajarewake(self):
        pattern = []
        for devided, preprocessed in zip(self.devided_pattern, self.preprocessed_pattern):
            if len(devided) == 0:
                pattern.append(False)
            elif self.list_max_dup(devided)[1] > 1 and self.sentence_max_dup_rate(self.list_max_dup(devided)[0], devided, preprocessed) <= 0.5:
                pattern.append(True)
            else:
                pattern.append(False)
        if any(pattern):
            return True
        # roman判定
        if any([self.roman_match(x) for x in self.roman_pattern]):
            return True
        return False

    def roman_match(self, sentence):
        """
        編集距離1までで単語のromanを探索
        keiki -> keeki
        sutoro -> sutero
        """
        words = {x for x in sentence.split(' ') if len(x) > self.min_roman}
        choices = ''.join(sentence.split(' '))
        for word in words:
            e = find_near_matches(word, choices.replace(word, ''), max_l_dist=1)
            if len(e) > 0:
                return True
        return False

    def list_max_dup(self, devided):
        return Counter(devided).most_common()[0]

    def sentence_max_dup_rate(self, sentence, devided, preprocessed):
        if self.kaburi == 2:
            return 0.0 if preprocessed.replace(sentence, '') else 1.0
        return devided.count(sentence) / len(set(devided))


if __name__ == '__main__':
    Shareka('これはテストです').dajarewake()
