from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger


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

    def load_model(self, path="word2vec_model.model"):
        """
        Load a trained model
        """
        self.model = Word2Vec.load(path)
        logger.info(f"Model loaded from {path}.")

    def get_embedding(self, text):
        """
        Get embedding for the given text
        """
        words = text.split()
        valid_words = [word for word in words if word in self.model.wv]
        if not valid_words:
            return None
        return np.mean([self.model.wv[word] for word in valid_words], axis=0)

    def recommend_tags(self, search_query, tags, top_n=10):
        """
        Recommend top N tags for the given search query
        """
        query_embedding = self.get_embedding(search_query)
        if query_embedding is None:
            logger.error("Query embedding could not be generated.")
            return []

        tag_embeddings = np.array([self.get_embedding(tag) for tag in tags if self.get_embedding(tag) is not None])
        tags = [tag for tag in tags if self.get_embedding(tag) is not None]

        if len(tag_embeddings) == 0:
            logger.error("No valid tag embeddings were found")
            return []

        similarities = cosine_similarity([query_embedding], tag_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return [tags[i] for i in top_indices]
