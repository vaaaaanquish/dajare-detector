#!/bin/bash

set -euC

poetry run python main.py dajare.EvalBertFeature --local-scheduler --rerun
poetry run python main.py dajare.EvalMultipleFeature --local-scheduler --rerun
poetry run python main.py dajare.EvalBertFeatureSampling --local-scheduler --rerun
poetry run python main.py dajare.EvalMultipleFeatureSampling --local-scheduler --rerun
