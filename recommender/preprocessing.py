import re


def preprocess_stopwords(tokens, stop_words):
    tokens = [
        token for token in tokens if token.lower() not in stop_words
        ]
    return tokens


def preprocess_input(df, column, token="word", stop_words=None):
    df = df.dropna().copy().reset_index(drop=True)
    if token == "word":
        df[column] = df[column]\
            .astype(str).str.lower()\
            .apply(lambda x: list(set(re.split(r"[,\s]+", x))))
    if token == "comb":
        df[column] = df[column].astype(str).str.lower().str.split(",")

    if stop_words:
        df[column] = df[column].apply(
            lambda x: preprocess_stopwords(x, stop_words)
            )
    return df


def prepare_train(*args):

    training_data = []
    for series in args:
        training_data.extend(series.tolist())

    return training_data


def get_uniques(tags):
    tags = tags.dropna()
    unique_tags = set(tag for tag_list in tags.tolist() for tag in tag_list)
    return unique_tags
