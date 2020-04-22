"""
Copyright © 2020 Johnson & Johnson
"""

import spacy


def get_spacy_nlp():
    try:
        spacy_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    except OSError:
        # We should tell the user explicitly what they need to do.
        raise Exception("Please run `python -m spacy download en_core_web_sm` locally.")

    return spacy_nlp
