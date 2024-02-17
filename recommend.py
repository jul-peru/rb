from dataloader import CSVDataLoader
from loguru import logger
from preprocessing import get_uniques, preprocess_input
from recommender import Word2VecTagRecommender



config = {
    "input_tags_path": "artist_tags.csv",
    "tags_column": "tags",
}

csv_loader = CSVDataLoader()
tags_column = config["tags_column"]

artists_tags = csv_loader.load_data(config["input_tags_path"], header=None, names=["tags"])
tags_processed = preprocess_input(artists_tags, tags_column, token="comb")

recommender = Word2VecTagRecommender()

model_path = "word2vec_artwork_search.model"
recommender.load_model(model_path)
unique_tags = get_uniques(tags_processed[tags_column])

search_query = "pink pokemon"
recommended_tags = recommender.recommend_tags(search_query, unique_tags)
logger.info("Recommended Tags: {}", recommended_tags)

# # print("initialising models")
# print("""
# {
#   "related_topics": ["sausage", "bun", "fast food"]
# }
# """)