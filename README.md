# dajare-detector
Japanese joke detection

ダジャレを検出するルールベース、機械学習モデル生成のためのスクリプト群です。

解説：ダジャレを判定する - Stimulator https://vaaaaaanquish.hatenablog.com/entry/2020/12/11/122721

# Usage

## train, eval

`./conf/param.ini`の以下を修正する
また、指定の`file_path`には、ダジャレデータをsampleと同じフォーマットで設置する
```
[dajare.LoadDajareData]
file_path=./data/dajare_sample.csv
```

モデルの学習と評価
```
docker build -t dajare .
docker run -it -v ${PWD}/resource:/app/resource dajare /app/batch/run_all_eval.sh
```

## twitter
`./conf/twitter_conf.yml`にTwitterのAPI情報を書く。

### ツイートデータの収集
```
python twitter/twitter_crawler.py
```

### モデル評価
```
docker build -t dajare .
docker run -it -v ${PWD}/resource:/app/resource dajare /app/batch/run_twitter_eval.sh
```
