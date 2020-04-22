"""
Copyright © 2020 Johnson & Johnson
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize_text(text_col: pd.Series,
                   vec_type: str = 'count',
                   **kwargs):
    """
    Vectorizes pre-processed text. Instantiates the vectorizer and
    fit_transform it to the data provided.

    :param text_col: Pandas series, containing preprocessed text.
    :param vec_type: string indicating what type of vectorization
        (count or tfidf currently).
    :param **kwargs: dict of keyworded arguments for sklearn vectorizer
        functions.

    :return: A tuple containing vectorized (doc-feature matrix that as d rows
        and f columns for count and tfidf vectorization) and vectorizer_obj
        (vectorization sklearn object representing trained vectorizer).
    """

    # Check if vectorization type is supported
    assert vec_type in ['count', 'tfidf']

    # Get raw values from pandas series
    text_raw = text_col.tolist()

    # Lets the vectorizer know the input has already been pre-tokenized
    # and is now delimited by whitespaces
    kwargs['analyzer'] = str.split

    # Apply proper vectorization
    if vec_type == 'count':
        count_vec = CountVectorizer(**kwargs)
        vectorized = count_vec.fit_transform(text_raw)
        vectorizer_obj = count_vec
    elif vec_type == 'tfidf':
        tfidf_vec = TfidfVectorizer(**kwargs)
        vectorized = tfidf_vec.fit_transform(text_raw)
        vectorizer_obj = tfidf_vec

    # Return vectorized object
    return vectorized, vectorizer_obj


def vectorize_new_text(text_col: pd.Series,
                       vectorizer_obj):
    """
    Vectorizes pre-processed new text. Used the provided vectorizer and
    apply/transform it to the data provided.

    :param text_col: -- Pandas series, containing preprocessed text.
    :param vectorizer_obj: -- Trained vectorizer object from vectorizing old text.

    :return: doc-feature matrix that has d rows and f columns for count and
        tfidf vectorization
    """

    # Check vectorization object
    assert type(vectorizer_obj) is CountVectorizer or TfidfVectorizer

    # Get raw values from pandas series
    text_raw = text_col.tolist()

    # Apply proper vectorization
    vectorized = vectorizer_obj.transform(text_raw)

    # Return vectorized object
    return vectorized
