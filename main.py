'''
pip install nltk
pip install scikit-learn
pip install numpy
pip install pandas
'''

import os
import matplotlib.pyplot as plt
import numpy as np
from datagen import datagen
from save_load import save, load
from Detection import MLRK, SVM, RF, DTree, NaiveBayes, KNN
from plot_result import plotres
from I_SBOA import I_SBOA, fit_func

# Ensure required directories exist
os.makedirs('Data Visualization', exist_ok=True)
os.makedirs('Results', exist_ok=True)
os.makedirs('Saved Data', exist_ok=True)

def full_analysis():
    # Check and generate data if not already saved
    if not os.path.exists('Saved Data/x_train_70.pkl'):
        print("Data not found. Generating data...")
        datagen()

    # Load 70% training, 30% testing data
    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')

    # Load 80% training, 20% testing data
    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    training_data = [
        (x_train_70, y_train_70, x_test_70, y_test_70),
        (x_train_80, y_train_80, x_test_80, y_test_80),
    ]

    for i, train_data in zip([70, 80], training_data):
        x_train, y_train, x_test, y_test = train_data

        # Feature selection
        lb = np.zeros(x_train.shape[1])
        ub = np.ones(x_train.shape[1])
        pop_size = 5
        prob_size = len(lb)
        epochs = 100

        best_solution, best_fitness = I_SBOA(fit_func, lb, ub, pop_size, prob_size, epochs, i)
        soln = np.round(best_solution)
        selected_indices = np.where(soln == 1)[0]

        # Ensure at least some features are selected
        if len(selected_indices) == 0:
            selected_indices = np.arange(x_train.shape[1])

        X_train = x_train[:, selected_indices]
        X_test = x_test[:, selected_indices]

        # Run and save results for each model
        for model_name, model_func in [
            ('proposed', MLRK),
            ('svm', SVM),
            ('naive_Bayes', NaiveBayes),
            ('rf', RF),
            ('dtree', DTree),
            ('knn', KNN),
        ]:
            pred, met = model_func(X_train, y_train, X_test, y_test)
            save(f'{model_name}_{i}', met)
            if model_name == 'proposed':  # Save predictions for the proposed model
                save(f'predicted_{i}', pred)

# Run the analysis
if __name__ == "__main__":
    full_analysis()
    plotres()
    plt.show()
