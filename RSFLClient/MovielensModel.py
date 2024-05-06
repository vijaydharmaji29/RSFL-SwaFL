from typing import Dict, Text
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from tensorflow.keras import Sequential, layers, Model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
tf.get_logger().setLevel('ERROR')

# model class
class RankingModel(Model):

    def __init__(self, unique_user_ids, unique_movie_titles, embedding_dim):
        super(RankingModel, self).__init__()
        self.userEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            layers.Embedding(input_dim=len(unique_user_ids) + 1, output_dim=embedding_dim)
        ])
        self.movieEmbedding = Sequential([
            layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            layers.Embedding(input_dim=len(unique_movie_titles) + 1, output_dim=embedding_dim)
        ])
        self.MLP = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1),
        ])

    def call(self, inputs):
        userID, movie_title = inputs
        userEmbedding = self.userEmbedding(userID)
        movieEmbedding = self.movieEmbedding(movie_title)
        output = self.MLP(tf.concat([userEmbedding, movieEmbedding], axis=-1))
        return output


class MovielensModel(tfrs.models.Model):

  def __init__(self, unique_user_ids, unique_movie_titles, embedding_dim):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel(unique_user_ids, unique_movie_titles, embedding_dim)
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model((features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")
    rating_predictions = self(features)
    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)