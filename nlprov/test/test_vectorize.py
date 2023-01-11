"""
Copyright © 2020 Johnson & Johnson
"""

import pytest
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nlprov.vectorize import vectorize_text, vectorize_new_text
from numpy import allclose


@pytest.fixture
def vectorize_actual():
    return pd.Series(['red dogs', 'red cats'])


@pytest.fixture
def vocab_set_expected(vectorize_actual):
    text_lol = vectorize_actual.str.split(' ').tolist()

    # Flattening list of lists via https://stackoverflow.com/a/11264751
    tokens = [val for sublist in text_lol for val in sublist]

    return set(tokens)


# Test for count vectorizer
@pytest.fixture
def count_dfm_expected():
    return csr_matrix([[0, 1, 1], [1, 0, 1]])


def test_count_vectorizer(vectorize_actual, count_dfm_expected,
                          vocab_set_expected):
    dfm, vec_obj = vectorize_text(vectorize_actual, vec_type='count')

    # Check sparse matrix is the same (or close enough)
    assert allclose(dfm.toarray(), count_dfm_expected.toarray())

    # Check vectorizer object type
    assert type(vec_obj) is CountVectorizer

    # Check original terms are included
    vocab_set = set(vec_obj.get_feature_names_out())
    assert vocab_set == vocab_set_expected


# Test for tfidf vectorizer
@pytest.fixture
def tfidf_dfm_expected():
    return csr_matrix([[0, 0.81480247, 0.57973867],
                       [0.81480247, 0, 0.57973867]])


def test_tfidf_vectorizer(vectorize_actual, tfidf_dfm_expected,
                          vocab_set_expected):
    dfm, vec_obj = vectorize_text(vectorize_actual, vec_type='tfidf')

    # Check sparse matrix is the same (or close enough)
    assert allclose(dfm.toarray(), tfidf_dfm_expected.toarray())

    # Check vectorizer object type
    assert type(vec_obj) is TfidfVectorizer

    # Check original terms are included
    vocab_set = set(vec_obj.get_feature_names_out())
    assert vocab_set == vocab_set_expected


# Test error of picking an invalid vec_type
def test_invalid_vec_type(vectorize_actual):
    with pytest.raises(Exception):
        vectorize_text(vectorize_actual, vec_type='Word2Vec')


# Test vectorization of new text
@pytest.fixture
def vectorizer_actual(vectorize_actual):
    _, vec_obj = vectorize_text(vectorize_actual, vec_type='count')
    return (vec_obj)


@pytest.fixture
def new_text_actual():
    return pd.Series(['blue cats'])


@pytest.fixture
def new_dfm_expected():
    return csr_matrix([[1, 0, 0]])


def test_vectorize_new(new_text_actual, vectorizer_actual,
                       new_dfm_expected):
    print(new_text_actual)
    new_dfm = vectorize_new_text(new_text_actual, vectorizer_actual)

    assert allclose(new_dfm.toarray(), new_dfm_expected.toarray())
