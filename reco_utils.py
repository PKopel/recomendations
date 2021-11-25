import math
import numpy as np
import pandas


K = 20
ALPHA = 0.00003
DELTA = 100
LAMBDA = 0.01
MAX_ITERATIONS = 100


def _initialize_users(raw_ratings, k):
    users_no = raw_ratings['userId'].unique().size
    users = pandas.DataFrame((5.0/k) * np.random.uniform(size=(users_no, k)), index=raw_ratings['userId'].unique(), columns=['x%s' % i for i in range(k)])
    users.sort_index(inplace=True) 
    return users_no, users


def _initialize_movies(raw_ratings, k):
    movies_no = raw_ratings['movieId'].unique().size
    movies = pandas.DataFrame(np.random.uniform(size=(movies_no, k)), index=raw_ratings['movieId'].unique(), columns=['x%s' % i for i in range(k)])
    movies.sort_index(inplace=True) 
    return movies_no, movies


def _get_ratings(raw_ratings):
    return raw_ratings.pivot(*raw_ratings.columns).fillna(0.0)


def _calculate_user_preferences(users, movies, ratings, raw_ratings, users_no, movies_no, alpha, delta, lambd, max_iter):
    total_error = 0.0
    iterations = 0
    users_model = users.copy()
    movies_model = movies.copy()
    
    while(True):
        previous_total_error = total_error

        predicted_ratings = users_model.dot(movies_model.T)
        errors = np.where(ratings==0.0, pandas.DataFrame(np.zeros((users_no, movies_no))), predicted_ratings - ratings)
        users_gradient = errors.dot(movies_model)
        movies_gradient = errors.T.dot(users_model)
        
        users_model = users_model - alpha * (users_gradient + lambd * users_model)
        movies_model = movies_model - alpha * (movies_gradient + lambd * movies_model)

        total_error = np.sum(errors ** 2)
        # print(total_error)
        progress = abs(previous_total_error - total_error)
        iterations += 1
        if progress < delta or iterations > max_iter:
            break
            
    return users_model, movies_model


def get_predicted_ratings(raw_ratings, k=K, alpha=ALPHA, delta=DELTA, lambd=LAMBDA, max_iterations=MAX_ITERATIONS):
    users_no, users = _initialize_users(raw_ratings, K)
    movies_no, movies = _initialize_movies(raw_ratings, K)
    ratings_from_dataset = _get_ratings(raw_ratings)
    users_model, movies_model = _calculate_user_preferences(users, movies, ratings_from_dataset, raw_ratings, users_no, movies_no, alpha, delta, lambd, max_iterations)
    
    ratings = 2 * users_model.dot(movies_model.T)
    return ratings.clip(0.0, 10.0).round().astype(int)
    