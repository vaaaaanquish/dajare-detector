#!/bin/bash

set -euC

poetry run python main.py dajare.EvalTwitterData --file-path=./data/tweet_data_under_sampling.csv --local-scheduler --rerun
poetry run python main.py dajare.EvalTwitterData --file-path=./data/tweet_data.csv --local-scheduler --rerun
