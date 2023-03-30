import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                   names=['user_id', 'item_id', 'rating', 'timestamp'])

folds = []
n_users = 943
n_items = 1682
for i in range(1, 6):
    train = pd.read_csv('ml-100k/u' + str(i) + '.base', sep='\t', header=None,
                        names=['user_id', 'item_id', 'rating', 'timestamp'])
    test = pd.read_csv('ml-100k/u' + str(i) + '.test', sep='\t', header=None,
                       names=['user_id', 'item_id', 'rating', 'timestamp'])
    ratings = np.zeros((n_users, n_items))

    for row in train.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]
    folds.append({'train': train, 'test': test, 'ratings': ratings})

# Define ALS function
def als(ratings, K=20, lambda_=0.1, n_iterations=10):
    m, n = ratings.shape
    U = np.random.rand(m, K)
    V = np.random.rand(n, K)
    for i in range(n_iterations):
        # Update user matrix
        for u in range(m):
            V_u = V[ratings[u, :] > 0, :]
            A = np.dot(V_u.T, V_u) + lambda_ * np.eye(K)
            b = np.dot(V_u.T, ratings[u, ratings[u, :] > 0])
            U[u, :] = np.linalg.solve(A, b)
        # Update item matrix
        for v in range(n):
            U_v = U[ratings[:, v] > 0, :]
            A = np.dot(U_v.T, U_v) + lambda_ * np.eye(K)
            b = np.dot(U_v.T, ratings[ratings[:, v] > 0, v])
            V[v, :] = np.linalg.solve(A, b)
    return U, V

def lfm(ratings, test_data, param):
    # Run ALS
    U, V = als(ratings, K=param[0], lambda_=param[1], n_iterations=param[2])

    # Impute missing values
    ratings_imputed = np.dot(U, V.T)

    y_true = test_data['rating'].values
    y_pred = ratings_imputed[test_data['user_id'] -
                             1, test_data['item_id'] - 1]

    errors = np.array(y_true) - np.array(y_pred)
    nmae = np.mean(np.abs(errors)) / np.mean(np.abs(y_true))
    return nmae


def cross_validation(folds, param):
    # Initialize the list of NMAEs
    nmaes = []

    # For each fold
    for fold in folds:
        # Compute the MAE for the algorithm and the fold
        nmae = lfm(fold['ratings'], fold['test'], param)

        # Add the MAE to the list of NMAEs
        nmaes.append(nmae)

    avg_nmae = np.mean(nmaes)
    nmaes.append(avg_nmae)

    # Return the list of NMAEs
    return nmaes

nmaes = []
params = [(10, 0.1, 10), (10, 0.1, 50), (20, 0.5, 20),
          (7, 0.5, 50), (10, 1, 10), (10, 0.1, 50), (20, 0.5, 20)]
for param in params:
    nmaes.append(cross_validation(folds, param))
    print(1)
    


nmaes = np.array(nmaes)
results = pd.DataFrame({'params(K, lambda, iterations)': params, 'Fold 1': nmaes[:, 0], 'Fold 2': nmaes[:, 1], 'Fold 3': nmaes[:,
                       2], 'Fold 4': nmaes[:, 3], 'Fold 5': nmaes[:, 4], 'Average': nmaes[:, 5]})
# results.set_index('params', inplace=True)
results

nmaes = []
params = [(10,1,50)]
for param in params:
    nmaes.append(cross_validation(folds, param))
    print(1)
nmaes