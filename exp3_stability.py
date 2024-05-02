import pandas as pd
import numpy as np
import surprise
from surprise import Reader, Dataset, SVD, GridSearchCV
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.special import softmax
np.random.seed(42)

# Display Python and matplotlib versions
import platform
print("Python version:", platform.python_version())
print("Matplotlib version:", plt.matplotlib.__version__)

# Suppress numpy warnings
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

# Data filtering based on genres
movies_df = pd.read_csv('../data/movies.csv')
no_genre_movies = movies_df[movies_df['genres'] == '(no genres listed)']['movieId'].tolist()
movies_df = movies_df[~movies_df['movieId'].isin(no_genre_movies)].reset_index(drop=True)
movies_df.drop(columns=['title'], axis=1, inplace=True)

data = pd.read_csv('../data/ratings.csv')
data = data[~data['movieId'].isin(no_genre_movies)].reset_index(drop=True)
data.drop(columns=['timestamp'], axis=1, inplace=True)

# Selecting top movies
top_movies = data.groupby('movieId', as_index=False)['rating'].sum().sort_values('rating', ascending=False)
top_movies = top_movies.head(100)['movieId'].tolist()
data = data[data['movieId'].isin(top_movies)]

# Data split
train_df = data.sample(frac=0.5, random_state=42)
test_df = data.drop(train_df.index)

# Model training and evaluation
reader = Reader(rating_scale=(0.5, 5))
train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
trainset = train_data.build_full_trainset()

param_grid = {'n_factors': [5, 50, 100], 'n_epochs': [5, 10], 'lr_all': [0.002, 0.005], 'reg_all': [0.02, 0.1]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(train_data)

print("Best RMSE:", gs.best_score['rmse'])
print("Best parameters:", gs.best_params['rmse'])

# Apply the best algorithm
algo = gs.best_estimator['rmse']
algo.fit(trainset)

# Prediction on the test set
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Visualization of results
results_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
results_df['err'] = abs(results_df.est - results_df.r_ui)
worst_predictions = results_df.sort_values('err', ascending=False).head(10)
print(worst_predictions)
