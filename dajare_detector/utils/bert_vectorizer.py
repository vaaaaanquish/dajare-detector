from logging import getLogger
import tarfile
import os

import requests
from sentence_transformers import SentenceTransformer

logger = getLogger(__name__)
"""
[require]

pip install git+https://github.com/sonoisa/sentence-transformers
"""


def load_sentence_bert(d):
    url = "https://www.floydhub.com/api/v1/resources/JLTtbaaK5dprnxoJtUbBbi?content=true&download=true&rename=sonobe-datasets-sentence-transformers-model-2"
    model_path = os.path.abspath(os.path.join(d, 'training_bert_japanese.tar'))

    if not os.path.exists(model_path):
        logger.info('download bert moodel')
        with open(model_path, 'wb') as f:
            for chunk in requests.get(
                    url, stream=True).iter_content(chunk_size=1024):
                f.write(chunk)
        tarfile.open(model_path).extractall(d)
    model = SentenceTransformer(model_path.strip('.tar'),
                                show_progress_bar=False)
    return model
