from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import get_uniques
import numpy as np
from loguru import logger
from collections import Counter


class Word2VecTagRecommender:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=1, sg=1, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.seed = seed
        self.model = None

    def train_model(self, sentences):
        """
        Train a Word2Vec model
        """
        self.model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, workers=self.workers, sg=self.sg)
        logger.info("Model training completed.")

    def save_model(self, path="word2vec_model.model"):
        """
        Save a trained model
        """
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}.")
        else:
            logger.warning("No model to save.")

    def load_model(self, path=None):
        """
        Load a trained model
        """
        if path is None:
            logger.error("Please, specify the path to load the model")
        self.model = Word2Vec.load(path)
        logger.info(f"Model loaded from {path}.")

    def get_embedding(self, text=None):
        """
        Get embedding for the given text
        """
        if text is None:
            logger.error("There is no text to make embeddings")
        words = text.split()
        valid_words = [word for word in words if word in self.model.wv]
        if not valid_words:
            return None
        return np.mean([self.model.wv[word] for word in valid_words], axis=0)

    def get_popular(self, tags, top_n=3):
        all_tags = [tag for tags_list in tags for tag in tags_list]
        tag_counts = Counter(all_tags)
        top_tags = tag_counts.most_common(top_n)
        top_tags_list = [tag for tag, _ in top_tags]
        return top_tags_list

    def recommend_tags(self, search_query, tags, top_n=3):
        """
        Recommend top N tags for the given search query
        """
        if top_n < 1:
            raise ValueError("Number of recommended tags must be at least 1")
        query_embedding = self.get_embedding(search_query)
        if query_embedding is None:
            logger.info("Query embedding could not be generated."
                        f"Returning top popular {top_n} tag(s)")
            return self.get_popular(tags, top_n=top_n)

        unique_tags = get_uniques(tags)
        tag_embeddings = np.array(
            [self.get_embedding(tag) for tag in
             unique_tags if self.get_embedding(tag) is not None]
             )
        unique_tags = [
            tag for tag in unique_tags if self.get_embedding(tag) is not None
            ]

        if len(tag_embeddings) == 0:
            logger.info("No valid tag embeddings were found"
                        f"Returning top popular {top_n} tag(s)")
            return self.get_popular(tags, top_n=top_n)

        similarities = cosine_similarity([query_embedding], tag_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return [unique_tags[i] for i in top_indices]
