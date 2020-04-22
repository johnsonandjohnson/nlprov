"""
Copyright © 2020 Johnson & Johnson
"""

import pandas as pd
from nlprov.preprocessing import preprocess_text
from nlprov.vectorize import vectorize_text, vectorize_new_text
from nlprov.similarity_calc import similarity_calculation

text = pd.Series(data=["  Combination  of   spaces.    ",
                       "MixEd CASe",
                       ",./;'[]\-=",
                       '<>?:"{}|_+',
                       '!@#$%^&*()`~"',
                       "lemmas needed",
                       "ducks and cats and ponies are not similar",
                       "c'est français",
                       "das ist deutsch",
                       "this is una mezcla"])
preprocessed_text = preprocess_text(text)
vec_text, vec_obj = vectorize_text(preprocessed_text)

new_text = pd.Series(data=["ducks and cats are not similar"])
new_preprocessed_text = preprocess_text(new_text)
new_vec_text = vectorize_new_text(new_preprocessed_text, vec_obj)

similarity = similarity_calculation(new_vec_text, vec_text)

