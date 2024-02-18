# Word2Vec Tag Recommender

## Overview

This Word2Vec Tag Recommender utilizes the Gensim Word2Vec model to generate semantic tag recommendations based on a given search query. The system is designed to initiate and train a model and recommend the most relevant tags by finding the clothest by the context and semantic meaning of words in the search query,.

## Features
NOTE: Please, initiate all scripts run from the `recommender` directory.
### Model initiation and training:
To initiate train of recommender model use 
```
python init.py --tags_file artist_tags.csv --search_file user_search_keywords.tsv
```
You will need to have a specific files to train it.
If files are called in this specific maner you can simply use:
```
python init.py
```

Trains a Word2Vec model according to Tags and Search queries.

### Semantic Tag Recommendations: 
To initiate please use:
```
python recommend.py --num_results 4 "your query"
```
--num_results should be between 1 and 15. Default parameter is 3.

Provides tag recommendations based on the semantic similarity between the search query and available tags.
Popular Tags: 
Offers popular tag recommendations if the search query contains words not present in the model's vocabulary.

Customizable Model Parameters: 
Allows adjusting Word2Vec model parameters such as vector size, window, and more. However, it should be changed directly in the init.py script, it is not passed as parameters now, but is scalable for future usage. 

Model Persistence: 
Supports saving and loading the trained Word2Vec model for reuse.
To make results of the model reproducable, please always use seeding and worker equal to 1.

## Setup

Prerequisites are in the requirements.txt file.