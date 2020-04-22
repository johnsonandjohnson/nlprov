import pandas as pd


def sents_chars_expected():
    return pd.Series(data=["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "abcdefghijklmnopqrstuvwxyz",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           ""
                           ])


def sents_nums_expected():
    return pd.Series(data=["",
                           "",
                           "0123456789",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           ""
                           ])


def sents_all_expected():
    return pd.Series(data=["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "abcdefghijklmnopqrstuvwxyz",
                           "0123456789",
                           "",
                           ",./;'[]\-=",
                           '<>?:"{}|_+',
                           '!@#$%^&*()`~"',
                           "Ω≈ç√∫˜µ≤≥÷",
                           "­؀؁؂؃؄؅؜۝܏᠎​‌‍‎‏‪",
                           "åß∂ƒ©˙∆˚¬…æ",
                           "œ∑´®†¥¨ˆøπ“‘",
                           "¡™£¢∞§¶•ªº–≠",
                           "¸˛Ç◊ı˜Â¯˘¿",
                           "ÅÍÎÏ˝ÓÔÒÚÆ☃",
                           "Œ„´‰ˇÁ¨ˆØ∏”’",
                           "`⁄€‹›ﬁﬂ‡°·‚—±",
                           "⅛⅜⅝⅞"
                           ])

