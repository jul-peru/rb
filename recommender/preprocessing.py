import re


def preprocess_stopwords(tokens, stop_words):
    """
    Removes stop words from a list of tokens

    Parameters
    ----------
    tokens : List[str]
        List of tokens to be preprocessed.
    stop_words : List[str]
        List of stop words to be removed from the tokens.

    Returns
    -------
    List[str]
        List of tokens after stop words removal.
    """
    tokens = [
        token for token in tokens if token.lower() not in stop_words
        ]
    return tokens


def preprocess_input(df, column, token="word", stop_words=None):
    """
    Preprocesses text data in a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the text data.
    column : str
        Name of the column containing text data.
    token : str
        Type of tokenization to apply.
        Either 'word' for word-level tokenization or
        'comb' for comma-separated tokenization.
        Default is 'word'
    stop_words : List[str] or None, optional
        List of stop words to remove from the text data. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with preprocessed text data.
    """
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
    """
    Prepares training data by combining multiple series into one list.

    Parameters
    ----------
    *args : pd.Series
        Series containing training data.

    Returns
    -------
    List[str]
        List containing combined training data.
    """
    training_data = []
    for series in args:
        training_data.extend(series.tolist())

    return training_data


def get_uniques(tags):
    """
    Extracts unique tags from a Series of tag lists.

    Parameters
    ----------
    tags : pd.Series
        Series containing lists of tags.

    Returns
    -------
    set
        Set containing unique tags.
    """
    tags = tags.dropna()
    unique_tags = set(tag for tag_list in tags.tolist() for tag in tag_list)
    return unique_tags
