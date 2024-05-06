from typing import Dict, Text
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from tensorflow.keras import Sequential, layers, Model
import os
import random
import math

from MovielensModel import MovielensModel


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
tf.get_logger().setLevel('ERROR')

class TrainingManager:
    def __init__(self, dataManager):
        self.ratings = dataManager.ratings
        self.model_weights = None
        self.model = MovielensModel(dataManager.userIDs, dataManager.movie_titles, 32)
        self.model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        self.dataManager = dataManager

    def testAccuracy(self):
        print("Measuring accuracy")
        accuracy_count = 0
        testDataset = list(self.ratings)
        random.shuffle(testDataset)
        testDataset = testDataset[:100]
        loaded = self.model

        loaded = tf.saved_model.load("./saved_model/")

        ctr = 0

        sqaured_sum = 0

        sqaured_sum_list = [0]*10

        for user in testDataset:
            ctr += 1
            user_id = user["user_id"].numpy().decode('utf-8')
            movie_title = user["movie_title"].numpy().decode('utf-8')
            user_rating = user["user_rating"].numpy()
            prediction = loaded({"user_id": np.array([user_id]), "movie_title": [movie_title]}).numpy()

            predicted_rating = prediction[0][0]
            
            acc_difference = abs(user_rating - predicted_rating)

            if acc_difference < 1:
                accuracy_count += 1

            sqaured_sum += (acc_difference*acc_difference)
            
            user_cluster = self.dataManager.user_clusters[int(user_id)]

            sqaured_sum_list[user_cluster] += (acc_difference*acc_difference)
            
        return sqaured_sum_list
    
    def train(self, ratings):
        shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        train = shuffled.take(int(len(ratings)*0.8)).batch(1)
        test = shuffled.skip(int(len(ratings)*0.8)).take(int(len(ratings)*0.2)).batch(1)

        if self.model_weights != None:
            self.model.set_weights(self.model_weights)

        self.model.fit(train, epochs=3)
        self.model_weights = self.model.get_weights()

        evaluation = self.model.evaluate(test, return_dict=True)

        print(evaluation)

