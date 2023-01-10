"""
Copyright © 2020 Johnson & Johnson
"""

import pandas as pd
import re
from spacy.lang.en.stop_words import STOP_WORDS as stop_set
from spacy.tokens import Token
from nltk.stem import PorterStemmer
import langid

from nlprov import get_spacy_nlp

ps = PorterStemmer()


def ps_stem(token):
    return ps.stem(token.text)


Token.set_extension("stem", getter=ps_stem, force=True)


def preprocess_text(text: pd.Series,
                    lowercase: bool = True,
                    regex: str = '(?![A-Za-z0-9]).',
                    replace_dict: dict = {},
                    nan_handling: str = 'remove',
                    lemma: bool = False,
                    stem: bool = False,
                    token_list: bool = False,
                    eng_lang: bool = True,
                    stop_words: bool = False):
    """
    Preprocessing text by optionally lowercasing, applying a regex of
    characters to keep, removing extra whitespace, lemmatizing, stemming,
    dictionary replacing symbols and terms, removing non-english text, and
    removing stop words.

    :param text: Pandas Series of strings.
    :param lowercase: Whether or not to lowercase text.
    :param regex: Regular expression of characters to keep.
    :param replace_dict: An optional dictionary of symbols and terms to
        replace.
    :param nan_handling: A string indicating removal of NAs/NaNs ('remove') or
        what should replace them in the text.
    :param lemma: Whether or not to lemmatize. Default False.
    :param stem: Whether or not to stem. Default False.
    :param token_list: Whether or not to return a series of token lists or
        space separated tokens.
    :param eng_lang: Remove non-english responses. Default True.
    :param stop_words: False (default) will not drop stop words. If True, spaCy
        stopwords will be removed.

    :return: Pandas Series, preprocessed text.
    """
    nlp = get_spacy_nlp()

    if stem and lemma:
        raise Exception('stem and lemma cannot both be true')

    if nan_handling == 'remove':
        text = text.dropna()
    else:
        text = text.fillna(nan_handling)

    if lowercase:
        text = text.str.lower()

    text = text.apply(lambda s: re.sub(regex, ' ', s))

    text = text.str.strip()
    text = text.str.replace(r'\s+', ' ', regex=True)

    if eng_lang:
        text = text[text.apply(lambda t: langid.classify(t)[0]) == 'en']

    if stop_words:
        text = text.apply(lambda doc: ' '.join(
            [item for item in doc.split(' ') if item not in stop_set]))

    # Full pipeline
    if lemma or stem:
        text = pd.Series(nlp.pipe(text), index=text.index)
        for ind, val in text.iteritems():
            end_str = []
            for item in val:
                if lemma:
                    end_str.append(item.lemma_)
                elif stem:
                    end_str.append(item._.stem)

            text[ind] = end_str

        if not token_list:
            text = text.apply(lambda desc: ' '.join([item for item in desc]))
    else:
        if token_list:
            # Only tokenization?
            text = pd.Series([nlp.make_doc(t) for t in text], index=text.index)
            text = text.apply(lambda doc: [tok.text for tok in doc])

    dict_replace = {'\xa0': ' '}
    dict_replace.update(replace_dict)
    text = text.replace(dict_replace, regex=True)

    return text
