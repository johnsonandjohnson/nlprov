FROM continuumio/miniconda3 as build

RUN mkdir -p /opt/app
WORKDIR /opt/app

ADD dev_environment.yml .
ADD setup.py .
ADD README.md .
ADD conftest.py .

ENV PYTHONUNBUFFERED=1

RUN /bin/bash -c "echo '. ~/anaconda/etc/profile.d/conda.sh' >> ~/.bash_profile && conda env create -f dev_environment.yml"

COPY nlprov ./nlprov

RUN /bin/bash -c 'source activate nlp_env && pytest --junitxml=results.xml --cov=nlprov --cov-report xml nlprov/test'

FROM python:latest

RUN mkdir -p /opt/app
WORKDIR /opt/app

ADD dev_environment.yml .
ADD setup.py .
ADD README.md .
ADD conftest.py .
ADD nlp_example.py .

COPY nlprov ./nlprov

RUN python -m pip install --user --upgrade setuptools wheel

RUN python setup.py sdist bdist_wheel

RUN python -m pip install --user dist/nlprov-1.1.0-py3-none-any.whl

RUN python -m spacy download en_core_web_sm

RUN python nlp_example.py