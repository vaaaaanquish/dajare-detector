"""
[require]
- $ pip install alkana.py pykakasi fuzzysearch mecab-python3
- $ pip install tweepy pandas neologdn emoji mojimoji regex tqdm
- $ pip install mecab==0.996.2
- $ cp conf/twitter_conf_sample.yml conf/twitter_conf.yml
- Install neologd
- setting twitter_conf.yml

[run]
python twitter_crawler.py
"""
import os
import yaml
import re

from dajare_detector import DajareDetector
import tweepy
import pandas as pd
import neologdn
import emoji
import mojimoji
import regex
import MeCab
from tqdm import tqdm

kanji_regex = regex.compile(r'\p{Script_Extensions=Han}+')


def load_file(directory, file_name, load_method):
    file_path = os.path.abspath(
        os.path.join(os.path.pardir, directory, file_name))
    with open(file_path, 'r') as f:
        data = load_method(f)
    return data


def normalize(text):
    text = text.replace('【', '「')
    text = text.replace('】', '」')
    text = text.replace('『', '「')
    text = text.replace('』', '」')
    text = re.sub(r'RT @[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]*?: ', '', text)
    text = re.sub(r'\u3000|\s', ' ', text)
    text = re.sub(r'#[^\s]+', '', text)
    text = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)",
                  '', text)
    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
    text = re.sub(r'@([A-Za-z0-9_]+) ', '', text)
    text = re.sub(r"pic.twitter.com/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]*",
                  '', text)
    text = text.lower()
    text = mojimoji.zen_to_han(text, kana=True)
    text = mojimoji.han_to_zen(text, digit=False, ascii=False)
    text = re.sub(r'\d+', '0', text)
    return neologdn.normalize(text)


def get_keyword(sentence, mecab):
    node = mecab.parseToNode(sentence)
    keywords = []
    while node:
        if node.feature.split(",")[0] == u"名詞":
            keywords.append(node.surface)
        elif node.feature.split(",")[0] == u"形容詞":
            keywords.append(node.feature.split(",")[6])
        elif node.feature.split(",")[0] == u"動詞":
            keywords.append(node.feature.split(",")[6])
        node = node.next
    return [
        word for word in keywords
        if len(word) > 1 or kanji_regex.fullmatch(word)
    ]


def twitter_search(word, api):
    q = f'{word} -RT lang:ja -filter:links -filter:retweets'
    results = api.search(q=q,
                         lang='ja',
                         tweet_mode='extended',
                         result_type='recent',
                         count=100)
    return {normalize(result.full_text) for result in results}


if __name__ == '__main__':
    config = load_file('config', 'twitter_config.yml', yaml.safe_load)
    df = load_file('data', 'dajare.csv', pd.read_csv)

    # parse
    mecab = MeCab.Tagger(f'-Ochasen -d {config["NEOLOGD"]}')
    word_set = {}
    for x in df['text']:
        word_set |= set(get_keyword(x, mecab))

    # twitter search
    # user認証よりapp認証の方が検索上限が高い
    # https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets
    auth = tweepy.AppAuthHandler(config['CONSUMER_KEY'],
                                 config['CONSUMER_SECRET'])
    api = tweepy.API(auth, wait_on_rate_limit=True)
    tweet_data = set()
    for x in tqdm(word_set):
        tweet_data |= twitter_search(x, api)

    # save all tweet
    file_path = os.path.abspath(
        os.path.join(os.path.pardir, 'data', 'tweet_data.csv'))
    df = pd.DataFrame({'text': list(tweet_data)})
    df.to_csv(file_path, index=False)

    # rule based under sampling
    dd = DajareDetector()
    df['rule_base'] = df['text'].apply(dd.do)
    file_path = os.path.abspath(
        os.path.join(os.path.pardir, 'data', 'tweet_data_under_sampling.csv'))
    pd.concat([df[df['dd']].sample(20000),
               df[~df['dd']].sample(20000)]).to_csv(file_path, index=False)
