import numpy as np
import pandas as pd
from recommender.recommender import Word2VecTagRecommender


sentences = [['apple', 'banana', 'orange'], ['apple', 'orange'], ['banana', 'kiwi']]


def test_train_model():
    recommender = Word2VecTagRecommender()
    recommender.train_model(sentences)
    assert recommender.model is not None


def test_load_model():
    recommender = Word2VecTagRecommender()
    recommender.load_model(path="tests/test_word2vec.model")
    assert recommender.model is not None


def test_get_embedding():
    recommender = Word2VecTagRecommender()
    recommender.train_model(sentences)
    embedding = recommender.get_embedding("apple orange")
    assert isinstance(embedding, np.ndarray)


def test_recommend_tags():
    recommender = Word2VecTagRecommender()
    recommender.train_model(sentences)
    tags = pd.Series([['apple', 'banana'], ['banana', 'kiwi']])
    recommended_tags = recommender.recommend_tags("apple", tags)
    print(tags)
    assert len(recommended_tags) > 0
