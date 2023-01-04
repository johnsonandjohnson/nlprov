"""
Copyright © 2020 Johnson & Johnson
"""

import en_core_web_sm


def get_spacy_nlp():
    try:
        spacy_nlp = en_core_web_sm.load(disable=['parser', 'ner', 'textcat'])
    except OSError:
        # We should tell the user explicitly what they need to do.
        raise Exception("Please run `python -m spacy download en_core_web_sm` locally.")

    return spacy_nlp
