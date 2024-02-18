"""
Recommendation pipeline.
"""
from dataloader import CSVDataLoader
# from loguru import logger
from preprocessing import preprocess_input
from recommender import Word2VecTagRecommender
import argparse
import json


parser = argparse.ArgumentParser(
    description="Recommend tags based on a search query"
    )
parser.add_argument(
    "--num_results", type=int, default=3,
    help="Number of recommended tags (default: 3). Set value between 1 and 15")
parser.add_argument("search_query", type=str, help="Search query")

# TODO Move config to a separate file
config = {
    "input_tags_path": "artist_tags.csv",
    "tags_column": "tags",
    "output_file": "results.json",
    "model_path": "word2vec_artwork_search.model",
}


def main(config):
    args = parser.parse_args()
    num_results = args.num_results
    search_query = args.search_query
    if (num_results < 1) | (num_results > 15):
        raise ValueError("Number of recommended tags must be between 1 and 15")

    csv_loader = CSVDataLoader()
    tags_column = config["tags_column"]
    artists_tags = csv_loader.load_data(
        config["input_tags_path"], header=None, names=[config["tags_column"]]
        )
    tags_processed = preprocess_input(artists_tags, tags_column, token="comb")
    recommender = Word2VecTagRecommender()
    model_path = config["model_path"]
    recommender.load_model(model_path)

    recommended_tags = recommender.recommend_tags(
        search_query, tags_processed[tags_column], top_n=num_results
        )

    result = {"related_topics": recommended_tags}
    with open(config["output_file"], "w") as f:
        json.dump(result, f)
    print(result)


if __name__ == "__main__":
    main(config)
