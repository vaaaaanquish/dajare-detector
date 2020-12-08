# dajare-detector
Japanese joke detection


# Usage

## train
```
# ダジャレデータをsampleと同じフォーマットで設置する
[dajare.LoadDajareData]
file_path=./data/dajare_sample.csv
```

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
