import MeCab
from collections import Counter
import re
import alkana


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
        mecab = MeCab.Tagger("-Oyomi -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        self.kana = mecab.parse(sentence)[:-1]
        self.preprocessed = self.preprocessing(self.kana)
        self.devided = self.devide(self.preprocessed)

    def preprocessing(self, sentence):
        """kana変換を追加"""
        alpha_dict = {x: alkana.get_kana(x.lower()) for x in self.alpha_regex.findall(sentence)}
        for k, v in alpha_dict.items():
            sentence = sentence.replace(k, v) if v else sentence

        for i, replace_word in enumerate(self.replace_words):
            sentence = sentence.replace(replace_word[0], replace_word[1])
        return sentence

    def devide(self, sentence):
        elements = []
        repeat_num = len(sentence) - (self.kaburi - 1)
        for i in range(repeat_num):
            elements.append(sentence[i:i + self.kaburi])
        return elements

    def dajarewake(self):
        if len(self.devided) == 0:
            return False
        elif self.list_max_dup()[1] > 1 and self.sentence_max_dup_rate(self.list_max_dup()[0]) <= 0.5:
            return True
        else:
            return False

    def list_max_dup(self):
        return Counter(self.devided).most_common()[0]

    def sentence_max_dup_rate(self, sentence):
        if self.kaburi == 2:
            return 0.0 if self.preprocessed.replace(sentence, '') else 1.0
        return self.devided.count(sentence) / len(set(self.devided))


if __name__ == '__main__':
    Shareka('これはテストです').dajarewake()
