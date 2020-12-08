FROM python:3.7.0-stretch

WORKDIR /app

RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all

# mecab
RUN apt install -y mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 gsutil libwww-perl
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH /root/google-cloud-sdk/bin:$PATH
RUN mkdir -p /usr/lib/x86_64-linux-gnu/mecab
RUN ln -s /var/lib/mecab/dic /usr/lib/x86_64-linux-gnu/mecab/dic

# python
WORKDIR /app
RUN pip install --upgrade pip &&\
    rm -rf ~/.cache
RUN pip install poetry
COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock
RUN poetry install

# files
WORKDIR /
COPY ./conf /app/conf
COPY ./data /app/data
COPY ./dajare_detector /app/dajare_detector
COPY ./main.py /app/main.py
COPY ./batch /app/batch

WORKDIR /app
ENTRYPOINT [ "/bin/bash" ]
VOLUME "/app"
