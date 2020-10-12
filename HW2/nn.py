import numpy as np
import matplotlib.pyplot as plt

from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import time

from mlrose.algorithms.decay import ExpDecay
from mlrose.neural import NeuralNetwork

from sklearn.metrics import log_loss, classification_report
from sklearn.model_selection import train_test_split


X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0 ,stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


iteration_marks = [10, 100, 250, 500, 1000, 2500, 5000]

def get_rhc(iteration_val):
	rhc = NeuralNetwork(hidden_nodes=[10], activation='relu',
                       algorithm='random_hill_climb', max_iters=iteration_val,
                       bias=True, is_classifier=True, learning_rate=0.001,
                       early_stopping=False, clip_max=1e10,
                       max_attempts=iteration_val, random_state=[100], curve=False)
	rhc.fit(X_train, y_train)
	pred = rhc.predict(X_test)
	acc = accuracy_score(y_test, pred) * 100
	return (acc)

def get_sa(iteration_val, cooling):
	exp_decay = ExpDecay(init_temp=100,
	                     exp_const=cooling,
	                     min_temp=0.001)

	sa = NeuralNetwork(hidden_nodes=[10], activation='relu',
	                      algorithm='simulated_annealing', max_iters=iteration_val,
	                      bias=True, is_classifier=True, learning_rate=0.001,
	                      early_stopping=False, clip_max=1e10, schedule=exp_decay,
                      max_attempts=iteration_val, random_state=[100], curve=False)
	sa.fit(X_train, y_train)
	pred = sa.predict(X_test)
	acc = accuracy_score(y_test, pred) * 100
	return (acc)

def get_ga(iteration_val, population, mutate):

	ga = NeuralNetwork(hidden_nodes=[10], activation='relu',
	                      algorithm='genetic_alg', max_iters=iteration_val,
	                      bias=True, is_classifier=True, learning_rate=0.001,
	                      early_stopping=False, clip_max=1e10,
	                      pop_size=population, mutation_prob=mutate,
	                      max_attempts=iteration_val, random_state=[100], curve=False)

	ga.fit(X_train, y_train)
	pred = ga.predict(X_test)
	acc = accuracy_score(y_test, pred) * 100
	return (acc)

rhc_scores = []
sa_scores = []
ga_scores = []

for i in iteration_marks:
	print(i)
	rhc_scores.append(get_rhc(i))
	temp_score_sa = []
	for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
		temp_score_sa.append((get_sa(i, CE), CE))
	temp_vals = sorted(temp_score_sa, key=lambda x: x[0], reverse=True)
	print(temp_vals[0])
	sa_scores.append(temp_vals[0][0])
	temp_score_ga = []
	for j in [10, 20, 50, 200, 500]:
		for k in [0.1, 0.2, 0.5, 0.75, 0.9]:
			temp_score_ga.append((get_ga(i, j, k), j, k))
	temp_vals = sorted(temp_score_ga, key=lambda x: x[0], reverse=True)
	print(temp_vals[0])
	ga_scores.append(temp_vals[0][0])

print(pd.DataFrame({"iteration_mark":iteration_marks, "rhc":rhc_scores, "sa":sa_scores, "ga":ga_scores}))


