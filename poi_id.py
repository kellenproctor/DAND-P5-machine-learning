#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# First elimination:
# Removed 'loan advances', 'director_fees', 'restricted_stock_deferred',
# and their associated fraction features for having no data for more than 80%
# of total entries
# After feature selection (see jupyter notebook for reference), I subbed
# bonus_fraction for bonus, and implemented the following final feature_list.
# The top 2, 4, 5, and 10 features will be used with SelectKBest for choosing
# an algorithm below.
features_list = ['poi',
    'salary',
    #'bonus',
    'bonus_fraction', #subbed for bonus
    'long_term_incentive',
    'deferred_income',
    'deferral_payments',
    'other',
    'expenses',
    'total_payments',

    'exercised_stock_options',
    'restricted_stock',
    'total_stock_value',

    'from_this_person_to_poi',
    'to_messages',
    'from_poi_to_this_person',
    'from_messages',
    'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
del data_dict["TOTAL"] # Based on mini-project
del data_dict["THE TRAVEL AGENCY IN THE PARK"] # Based on looking at names
del data_dict["LOCKHART EUGENE E"] # Because he has no data


### Task 3: Create new feature(s)

# Create features list for easy creation of new features
payment_features = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees']

stock_features = ['exercised_stock_options', 'restricted_stock',
    'restricted_stock_deferred']

# Create new features
for p in data_dict:
    # Create fractional value of emails to POIs
    if data_dict[p]['from_this_person_to_poi'] == "NaN" or \
                data_dict[p]['from_messages'] == "NaN":
        data_dict[p]['to_poi_fraction'] = "NaN"
    else:
        data_dict[p]['to_poi_fraction'] = \
                data_dict[p]['from_this_person_to_poi'] / \
                float(data_dict[p]['from_messages'])

    # Create fractional value of emails from POIs
    if data_dict[p]['from_poi_to_this_person'] == "NaN" or \
                    data_dict[p]['to_messages'] == "NaN":
        data_dict[p]['from_poi_fraction'] = "NaN"
    else:
        data_dict[p]['from_poi_fraction'] = \
                data_dict[p]['from_poi_to_this_person'] / \
                float(data_dict[p]['to_messages'])

    for pay in payment_features:
        if data_dict[p][pay] == "NaN" or data_dict[p]['total_payments'] == "NaN":
            data_dict[p][pay + "_fraction"] = "NaN"
        else:
            data_dict[p][pay + "_fraction"] = float(data_dict[p][pay]) / \
                                            data_dict[p]['total_payments']

    for stock in stock_features:
        if data_dict[p][stock] == "NaN" or \
                data_dict[p]['total_stock_value'] == "NaN":
            data_dict[p][stock + "_fraction"] = "NaN"
        else:
            data_dict[p][stock + "_fraction"] = float(data_dict[p][stock]) / \
                                            data_dict[p]['total_stock_value']


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline, Pipeline

# Provided to give you a starting point. Try a variety of classifiers.
# Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gauss_clf = GaussianNB()

# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(min_samples_split=10)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
ranfor_clf = RandomForestClassifier()

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
adab_clf = AdaBoostClassifier()

# Create pipeline then run test of resulting algos
anova_filter = SelectKBest(f_classif, k=10)
algo_list = [gauss_clf, tree_clf, ranfor_clf, adab_clf]

# Create pipe function and run fit on algorithms with 10 SelectKBest
def pipe_and_report(algo, labels, features):
    pipe = make_pipeline(anova_filter, algo)
    pipe.fit(features, labels)
    pred = pipe.predict(features)
    clf_report = classification_report(labels, pred)
    #print clf_report

#for algo in algo_list:
    #print algo
    #pipe_and_report(algo, labels, features)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# import GridSearchCV and make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

# set up the SelectKBest feature selection and Random Forest algorithm
anova_filter = SelectKBest(f_classif, k=5)

ranfor_clf = RandomForestClassifier(max_features=None,
                                    n_estimators=20,
                                    criterion='gini')

pipe = make_pipeline(anova_filter, ranfor_clf)

# Accuracy
scorer = make_scorer(accuracy_score)

# Setup parameters for GridSearchCV tuning
params = {'selectkbest__k':[2, 4, 5, 6, 8, 10, 12],
         'randomforestclassifier__n_estimators':[2, 4, 5, 8, 10, 15, 20, 25, 50, 100],
         'randomforestclassifier__max_features':["auto", None],
         'randomforestclassifier__criterion':['gini', 'entropy']}

gridsearch = GridSearchCV(pipe, params, scoring=scorer)

# Run gridsearch on basic testing data
gridsearch.fit(features_train, labels_train)

# And our best estimator
#print "Best Estimator"
#print gridsearch.best_params_

#print "Top Three Options:"
#print sorted(gridsearch.grid_scores_, key=lambda score: score[1], reverse=True)[:3]

# Classification Report
#pred = gridsearch.predict(features_test)
#clf_report = classification_report(labels_test, pred)
#print clf_report


# Final configuration
anova_filter = SelectKBest(f_classif, k=5)
gauss_clf = GaussianNB()

pipe = make_pipeline(anova_filter, gauss_clf)

clf = pipe
clf = clf.fit(features_train, labels_train)

# Metrics
from sklearn.metrics import recall_score, precision_score, accuracy_score

def score(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print clf
    print "Score:", accuracy_score(y_test, pred)*100
    print "Recall:", recall_score(y_test, pred)*100
    print "Precision:", precision_score(y_test, pred)*100

score(clf, features_train, features_test, labels_train, labels_test)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)