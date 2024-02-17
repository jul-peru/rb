import pandas as pd
from stop_words import get_stop_words
from dataloader import CSVDataLoader
from preprocessing import preprocess_input, prepare_train
from recommender import Word2VecTagRecommender
from loguru import logger

config = {
    "input_tags_path": "artist_tags.csv",
    "input_search_path": "user_search_keywords.tsv",
    "tags_column": "tags",
    "keywords_column": "keywords"
}

csv_loader = CSVDataLoader()
tags_column = config["tags_column"]
keywords_column = config["keywords_column"]

artists_tags = csv_loader.load_data(config["input_tags_path"], header=None, names=["tags"])
search_keywords = csv_loader.load_data(config["input_search_path"])
# print(artists_tags.head(1))
# print(search_keywords.head(1))

stop_words = get_stop_words("english")
search_processed = preprocess_input(search_keywords, keywords_column, token="comb", stop_words=stop_words)
tags_processed = preprocess_input(artists_tags, tags_column, token="comb")

print(search_processed.head(1))
print(tags_processed.head(1))

# training model
training_data = prepare_train(tags_processed[tags_column], search_processed[keywords_column])
print(training_data[0])

recommender = Word2VecTagRecommender()
recommender.train_model(training_data)
recommender.save_model("word2vec_artwork_search.model")


