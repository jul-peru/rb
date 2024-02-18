import pandas as pd
import pytest
from recommender.preprocessing import preprocess_stopwords, preprocess_input, prepare_train, get_uniques


test_data = pd.DataFrame({
    "keywords": [["this", "pokemon"], ["pokemon"], ["red rain"], ["python test", "python"], ["red bubble"]]
})
stop_words = {"this", "is", "a"}


def test_preprocess_stopwords():
    tokens = ["this", "python", "interesting", "stop"]
    expected_output = ["python", "interesting", "stop"]
    assert preprocess_stopwords(tokens, stop_words) == expected_output


def test_preprocess_input():
    input_df = test_data.copy()
    expected_output = ["pokemon"]
    input_df["keywords"] = input_df["keywords"].apply(lambda x: preprocess_stopwords(x, stop_words))
    print(input_df)
    assert input_df["keywords"][0] == expected_output


def test_prepare_train():
    data1 = pd.Series(["tag1", "tag2"])
    data2 = pd.Series(["tag3", "tag4"])
    expected_output = ["tag1", "tag2", "tag3", "tag4"]
    assert prepare_train(data1, data2) == expected_output


def test_get_uniques():
    tags = pd.Series([["tag1", "tag2"], ["tag3", "tag4"]])
    expected_output = {"tag1", "tag2", "tag3", "tag4"}
    assert get_uniques(tags) == expected_output
