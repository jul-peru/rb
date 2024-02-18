import argparse
import pandas as pd
from stop_words import get_stop_words
from dataloader import CSVDataLoader
from preprocessing import preprocess_input, prepare_train
from recommender import Word2VecTagRecommender
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Artwork Search Recommendation System")
    parser.add_argument("--tags_file", help="Path to the artist tags CSV file")
    parser.add_argument("--search_file", help="Path to the user search keywords TSV file")
    args = parser.parse_args()

    config = {
        "tags_column": "tags",
        "keywords_column": "keywords",
        "tokens": "comb",
        "path_model_save": "word2vec_artwork_search.model",
    }

    tags_file = args.tags_file
    search_file = args.search_file

    csv_loader = CSVDataLoader()
    tags_column = config["tags_column"]
    keywords_column = config["keywords_column"]

    artists_tags = csv_loader.load_data(tags_file, header=None, names=[config["tags_column"]])
    search_keywords = csv_loader.load_data(search_file)

    stop_words = get_stop_words("english")
    search_processed = preprocess_input(search_keywords, keywords_column, token=config["tokens"], stop_words=stop_words)
    tags_processed = preprocess_input(artists_tags, tags_column, token=config["tokens"])

    training_data = prepare_train(tags_processed[tags_column], search_processed[keywords_column])

    recommender = Word2VecTagRecommender()
    recommender.train_model(training_data)
    recommender.save_model(config["path_model_save"])

if __name__ == "__main__":
    main()
