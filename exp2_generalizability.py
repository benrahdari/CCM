import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from scipy import stats

# Load data
data = pd.read_csv("./data/ratings.csv")

# Define movie popularity
popularity_threshold = 50
movie_popularity = data['movieId'].value_counts()
popular_movies = movie_popularity[movie_popularity > popularity_threshold].index
data_popular = data[data['movieId'].isin(popular_movies)]

# Define active users
user_activity_threshold = 50
user_activity = data['userId'].value_counts()
active_users = user_activity[user_activity > user_activity_threshold].index
data_active = data_popular[data_popular['userId'].isin(active_users)]

# Train-test split
train_df, test_df = train_test_split(data_active, test_size=0.2, random_state=42)

# Save train and test datasets
train_df.to_csv('./files/train.csv')
test_df.to_csv('./files/test.csv')

# Print dataset info
print(f"Whole Dataset: \t Movies: {len(np.unique(data['movieId']))} \t Users: {len(np.unique(data['userId']))} \t Ratings: {len(data['rating'])}")
print(f"Training Dataset: \t Movies: {len(np.unique(train_df['movieId']))} \t Users: {len(np.unique(train_df['userId']))} \t Ratings: {len(train_df['rating'])}")
print(f"Testing Dataset: \t Movies: {len(np.unique(test_df['movieId']))} \t Users: {len(np.unique(test_df['userId']))} \t Ratings: {len(test_df['rating'])}")

# using matrix factorization to compute the user rating matrix
# Finding the best parameters
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader=reader)
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
param_grid = {'n_factors': [4, 6, 9, 11, 14, 18, 29]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
gs.fit(data)

# Best RMSE score
print(gs.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Using the best parameter to fit the SVD
algo = SVD(n_factors=gs.best_params['rmse']['n_factors'])
algo.fit(trainset)

# Predict ratings for all pairs (i, j) that are NOT in the training set.
predictions = algo.test(testset)

# Using matrix factorization to compute the user rating matrix
# Finding the best parameters
test_data = Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader=reader)
test_trainset = test_data.build_full_trainset()
test_testset = test_trainset.build_anti_testset()

# Statistical tests
def t_stat(l1, l2):
    t_statistic, p_value = stats.ttest_ind(l1, l2)
    w_statistic, p_value = stats.levene(l1, l2)
    
    print("Independent t-test:")
    print(f"t-statistic = {t_statistic}")
    print(f"p-value = {p_value}")

    print("\nLevene's test for homogeneity of variance:")
    print(f"W-statistic = {w_statistic}")
    print(f"p-value = {p_value}")

t_stat(train_df['rating'], test_df['rating'])

# Plotting
cpd = pd.read_csv('./data/cpd_data.csv')
cpd_line = cpd.sort_values('num_ratings')
x = cpd_line['num_ratings']
fig, ax = plt.subplots()

test_line = [cpd_line['cp_test'].mean()] * 610
train_line = [cpd_line['cp_train'].mean()] * 610
popular_line = [cpd_line['cp_popular'].mean()] * 610
random_line = [cpd_line['cp_random'].mean()] * 610

rects2 = ax.scatter(cpd_line['cp_train'], cpd_line['num_ratings'], label='Train Set', color='tab:blue')
rects3 = ax.scatter(cpd_line['cp_popular'], cpd_line['num_ratings'], label='Popularity', color='tab:orange')
rects4 = ax.scatter(cpd_line['cp_random'], cpd_line['num_ratings'], label='Random', color='tab:red')

ax.set_ylabel('Average Click Probability')
ax.set_xlabel('Number of Ratings')
ax.set_title('Comparison of Different Metrics')
ax.legend()

fig.tight_layout()
plt.show()
