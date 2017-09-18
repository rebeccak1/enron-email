#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import operator

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import (VarianceThreshold, f_classif, SelectKBest)

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import (svm, preprocessing)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):

    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']

    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
	ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

	ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
	ax.set_xlabel(name_param_1, fontsize=16)
	ax.set_ylabel('CV Average F1 Score', fontsize=16)
	ax.legend(loc="best", fontsize=15)
	ax.grid('on')
    #plt.show()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 

#email_address
email_features =  ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

POI_label = ['poi'] 

features_list = POI_label + financial_features + email_features # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

for person in my_dataset:
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['fraction_from_poi'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['fraction_from_poi'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['fraction_to_poi'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['fraction_to_poi'] = 0


new_features_list = features_list + ['fraction_to_poi', 'fraction_from_poi']

data = featureFormat(my_dataset, new_features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)
print "labels"
print labels

'''
optimized_features_list = features_list
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
'''
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=15)
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=16) 

models = {'LogisticRegression', 'SVC', 'GaussianNB'}

parameters = {'LogisticRegression': {'LogisticRegression__C':[1e7,1e8,1e9], 'kbest__k':range(1,20)},
	      'SVC': {'SVC__C': [0.1, 1, 10, 100, 1000],'kbest__k':range(1,20)},
	      'GaussianNB': {'kbest__k':range(1,20)},
	      }

pipes = {'SVC': Pipeline(steps=[('kbest',SelectKBest()),('SVC',svm.SVC(kernel='poly', gamma=1))]),
	 'GaussianNB': Pipeline(steps=[('kbest',SelectKBest()),('Gaussian NB',GaussianNB())]), 'LogisticRegression': Pipeline(steps=[('kbest',SelectKBest()),('LogisticRegression',LogisticRegression(tol=1e-6))])}

params_chosen = {}
for m in models:
    
    print m
    params_chosen[m] = []
    params = parameters[m]
    pipe = pipes[m]

    grid_search = GridSearchCV(pipe, params, cv=cv, scoring='f1').fit(features, labels)
    print("Average CV accuracy: {}".format(np.mean(cross_val_score(grid_search, features, labels, cv=sss, scoring='accuracy'))))
    print("Average CV precision: {}".format(np.mean(cross_val_score(grid_search, features, labels, cv=sss, scoring='precision'))))
    print("Average CV recall: {}".format(np.mean(cross_val_score(grid_search, features, labels, cv=sss, scoring='recall'))))

    if m=='LogisticRegression':
	plot_grid_search(grid_search.cv_results_, range(1,20), [1e7,1e8,1e9],'k','LogisticRegression__C')
    elif m=='SVC':
	plot_grid_search(grid_search.cv_results_, range(1,20), [0.1, 1, 10, 100, 1000],'k','SVC__C')
    else:
	plot_grid_search(grid_search.cv_results_, range(1,20), ['no params'],'k','GaussianNB')

    print 'best_params'
    best_params = grid_search.best_estimator_.get_params()
    best_params = grid_search.best_params_
    print best_params
    kbest = best_params['kbest__k']
    selector = grid_search.best_estimator_#.steps[0][1]
    select_indices = selector.named_steps['kbest'].scores_
    for i in sorted(zip(select_indices, features_list[1:]),key=operator.itemgetter(0),reverse=True)[:kbest]:
	print i
	params_chosen[m].append(i[1])



    for param_name in params.keys():
	print "{} = {}, ".format(param_name, best_params[param_name])
    print '\n'

clf = GaussianNB()
features_list = params_chosen['GaussianNB']
features_list = ['poi'] + features_list

### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
