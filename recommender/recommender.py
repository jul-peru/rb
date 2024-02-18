from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import get_uniques
import numpy as np
from loguru import logger
from collections import Counter


class Word2VecTagRecommender:
    """
    A class for recommending tags based on Word2Vec embeddings.

    Attributes
    ----------
    vector_size : int
        Dimensionality of the word vectors. Default is 100.
    window : int
        Maximum distance between the current and predicted word within a sentence. Default is 5.
    min_count : int
        Ignores all words with total frequency lower than this. Default is 1.
    workers : int
        Number of workers used for training. Default is 1.
    sg : int
        Training algorithm: 1 for skip-gram; otherwise CBOW. Default is 1.
    seed : int
        Seed for the random number generator. Default is 42.
    model : gensim.models.Word2Vec or None
        Trained Word2Vec model.

    Methods
    -------
    train_model(sentences)
        Train a Word2Vec model.
    save_model(path="word2vec_model.model")
        Save a trained model.
    load_model(path=None)
        Load a trained model.
    get_embedding(text=None)
        Get embedding for the given text.
    get_popular(tags, top_n=3)
        Get the top N popular tags from the given tags list.
    recommend_tags(search_query, tags, top_n=3)
        Recommend top N tags for the given search query.
    """
    def __init__(self, vector_size=100, window=5, min_count=1, workers=1, sg=1, seed=42):
        """
        Initialize an instance of Word2VecTagRecommender class.

        Parameters
        ----------
        vector_size : int, optional
            Dimensionality of the word vectors. Default is 100.
        window : int, optional
            Maximum distance between the current and predicted word. Default is 5.
        min_count : int, optional
            Ignores all words with total frequency lower than this. Default is 1.
        workers : int, optional
            Number of workers used for training. Default is 1.
        sg : int, optional
            Training algorithm: 1 for skip-gram; otherwise CBOW. Default is 1.
        seed : int, optional
            Seed for the random number generator. Default is 42.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.seed = seed
        self.model = None

    def train_model(self, sentences):
        """
        Trains a Word2Vec model.

        Parameters
        ----------
        sentences : list of list of str
            List of sentences (tokenized) to train the model.
        """
        self.model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, workers=self.workers, sg=self.sg)
        logger.info("Model training completed.")

    def save_model(self, path="word2vec_model.model"):
        """
        Save a trained model.

        Parameters
        ----------
        path : str, optional
            Path to save the model. Default is "word2vec_model.model".
        """
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}.")
        else:
            logger.warning("No model to save.")

    def load_model(self, path=None):
        """
        Load a trained model.

        Parameters
        ----------
        path : str or None, optional
            Path to load the model from. If None, an error is logged.
        """
        if path is None:
            logger.error("Please, specify the path to load the model")
        self.model = Word2Vec.load(path)
        logger.info(f"Model loaded from {path}.")

    def get_embedding(self, text=None):
        """
        Get embedding for the given text.

        Parameters
        ----------
        text : str or None, optional
            Text to get embedding for. If None, an error is logged.

        Returns
        -------
        numpy.ndarray or None
            Embedding vector for the given text, or None if text is None or no valid embedding found.
        """
        if text is None:
            logger.error("There is no text to make embeddings")
        words = text.split()
        valid_words = [word for word in words if word in self.model.wv]
        if not valid_words:
            return None
        return np.mean([self.model.wv[word] for word in valid_words], axis=0)

    def get_popular(self, tags, top_n=3):
        """
        Get the top N popular tags from the given tags list.

        Parameters
        ----------
        tags : list of list of str
            List of tags.
        top_n : int, optional
            Number of top tags to retrieve. Default is 3.

        Returns
        -------
        list of str
            List of top N popular tags.
        """
        all_tags = [tag for tags_list in tags for tag in tags_list]
        tag_counts = Counter(all_tags)
        top_tags = tag_counts.most_common(top_n)
        top_tags_list = [tag for tag, _ in top_tags]
        return top_tags_list

    def recommend_tags(self, search_query, tags, top_n=3):
        """
        Recommend top N tags for the given search query.

        Parameters
        ----------
        search_query : str
            Search query for which tags are to be recommended.
        tags : list of list of str
            List of tags.
        top_n : int, optional
            Number of top tags to recommend. Default is 3.

        Returns
        -------
        list of str
            List of recommended tags.
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
