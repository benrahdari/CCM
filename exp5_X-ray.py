import pandas as pd
import numpy as np
import surprise
from surprise import Reader, Dataset, SVD, GridSearchCV
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
from scipy.special import softmax
np.random.seed(42)
import random
from math import log
import ast

import platform
print("python %s" % platform.python_version())
print("matplotlib %s" % plt.matplotlib.__version__)

np.seterr(invalid='ignore')
np.seterr(divide='ignore')

def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    total = len(iterable)
    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    printProgressBar(0)
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()

# Load data
movies_df = pd.read_csv('../data/movies.csv')
no_genre_movies = movies_df[movies_df['genres'] == '(no genres listed)']['movieId'].tolist()
movies_df = movies_df[~movies_df['movieId'].isin(no_genre_movies)].reset_index(drop=True)
movies_df.drop(columns=['title'], inplace=True)

data = pd.read_csv('../data/ratings.csv')
data = data[~data['movieId'].isin(no_genre_movies)].reset_index(drop=True)
data.drop(columns=['timestamp'], inplace=True)

top_movies = data.groupby('movieId')['rating'].sum().sort_values(ascending=False).head(100).index.tolist()
data = data[data['movieId'].isin(top_movies)]

# Model setup
reader = Reader(rating_scale=(0.5, 5))
train_data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
trainset = train_data.build_full_trainset()
testset = trainset.build_anti_testset()

# Model training
param_grid = {'n_factors': [5, 50, 100], 'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.1]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(train_data)

print("Best RMSE:", gs.best_score['rmse'])
print("Best parameters:", gs.best_params['rmse'])

# Apply the best algorithm
algo = gs.best_estimator['rmse']
algo.fit(trainset)

# Prediction on the test set
predictions = algo.test(testset)

def getMovieAttractions(predictions, userId):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if uid == userId:
            top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings

    preds_df = pd.DataFrame([(id, pair[0], pair[1]) for id, row in top_n.items() for pair in row], columns=["userId", "movieId", "rating"])
    pred_usr = preds_df[preds_df["userId"] == userId].merge(movies_df, on='movieId')
    hist_usr = data[data.userId == userId].sort_values("rating", ascending=False).merge(movies_df, on='movieId')
    pred_all = pd.concat([pred_usr, hist_usr], axis=0)
    pred_all['rating'] = softmax(pred_all['rating'].tolist())
    return pred_all

# Example usage
user_attractions = getMovieAttractions(predictions, 1)
print("Mean attraction for user 1:", user_attractions['rating'].mean())
