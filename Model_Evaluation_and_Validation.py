import copy
import pickle
import winsound
from itertools import product
from random import sample

import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from Classifiers import classifiers, params, params_single
from HiggsBosonCompetition_AMSMetric_rev1 import AMS


def AMS_scoring(ground_truth=None, predictions=None, **kwargs):
    w = kwargs['w']
    ground_truth.reshape(-1, 1)
    predictions.reshape(-1, 1)
    s = np.sum(w * ground_truth * predictions)
    ground_truth_transf = np.array([1 if c == 0 else 0 for c in ground_truth])
    b = np.sum(w * ground_truth_transf * predictions)
    return AMS(s, b)


data_file_name = 'Datasets\data_mi_imp.pkl' #name of the data file to be used to build the learning model
is_initial = False # bool indicating wether to use default classifier params (False) or use grid search (True)
cla_choice = 'dt' # choice of classifier to use as set up in the Classifiers.py file

# unpickle data set
with open(data_file_name, 'rb') as f:
    pkl_data = pickle.load(f)

df = pkl_data['tr_data']

# ############## NORMALIZATION #################
rs = RobustScaler()
rs = rs.fit(df)
df.iloc[:] = rs.transform(df)
# ##############################################

data = df.values

# filter = sample(list(range(0, len(data))), len(data)//10)
# with open('ten_filter.pkl', 'wb') as f:
#     pickle.dump(dict(sample=filter), f)

with open('ten_filter.pkl', 'rb') as f: # saved filter that subsamples data (10%)
    sample = pickle.load(f)

filter = sample['sample']

X = np.array(data)[filter, :]
Y = np.array(pkl_data['lbls'].values)[filter]
weights = pkl_data['w'][filter]

kfold = KFold(n_splits=5, random_state=7)

cla = classifiers[cla_choice]
param_grid = product(*params[cla_choice].values())
if type(cla) in [BaggingClassifier, AdaBoostClassifier]:
    cla_choice = 'dt'
    param_grid = product(*params_single[cla_choice].values())

best_params_ams = []
best_result_ams = 0
best_cla_ams = None

best_params_acc = []
best_result_acc = 0
best_cla_acc = None

i = 0

# Grid search loop
for param_choice in param_grid:

    # extract params
    if type(cla) == BaggingClassifier:
        choices = {key: param_choice[i] for i, key in enumerate(params_single[cla_choice].keys())}
        Bagging_choices = {"base_estimator": classifiers[cla_choice].set_params(**choices), 'n_estimators': 20,
                           'max_samples': 1.0}
        if not is_initial:
            cla.set_params(**Bagging_choices)

    elif type(cla) == AdaBoostClassifier:
        choices = {key: param_choice[i] for i, key in enumerate(params_single[cla_choice].keys())}
        Ada_choices = {"base_estimator": classifiers[cla_choice].set_params(**choices), 'n_estimators': 50,
                       'learning_rate': 0.01, 'algorithm': 'SAMME.R'}
        if not is_initial:
            cla.set_params(**Ada_choices)

    else:
        choices = {key: param_choice[i] for i, key in enumerate(params[cla_choice].keys())}
        if not is_initial:
            cla.set_params(**choices)

    results_ams = []
    results_acc = []
    # CV loop
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        w_test = weights[test_index]
        scoring = make_scorer(AMS_scoring, greater_is_better=True, w=w_test)
        cla = cla.fit(X_train, Y_train)
        Y_pred = cla.predict(X_test)

        # print('ROC metric is {:0.2f} and accuracy is {:0.2f}'.format(roc_auc_score(Y_test, Y_pred),
        #                                                                accuracy_score(Y_test, Y_pred)))

        results_ams.append(scoring(cla, X_test, Y_test))
        results_acc.append(accuracy_score(Y_test, Y_pred))

    i += 1
    print(param_choice)
    print("\nAt grid point number %d, AMS/ACC score is %0.1f (+/- %0.1f) / %0.1f (+/- %0.1f)" %
          (i, np.mean(results_ams), np.std(results_ams), np.mean(results_acc)*100, np.std(results_acc)*100))

    if np.mean(results_ams) > best_result_ams:
        print("%0.3f IS BETTER THAN %0.03f" % (np.mean(results_ams), best_result_ams))
        best_result_ams = np.mean(results_ams)
        best_result_acc = np.mean(results_acc)
        print('Best params for AMS: {}'.format(param_choice))
        best_params_ams = copy.copy(param_choice)
        best_cla_ams = copy.copy(cla)

    # if np.mean(results_acc) > best_result_acc:
    #     print("%0.3f IS BETTER THAN %0.03f" % (np.mean(results_acc), best_result_acc))
    #     best_result_acc = np.mean(results_acc)
    #     print('Best params for ACC: {}'.format(param_choice))
    #     best_params_acc = copy.copy(param_choice)
    #     best_cla_acc = copy.copy(cla)

    if is_initial:
        break

print("\nBest AMS/ACC score is %0.1f / %0.1f" % (best_result_ams, best_result_acc*100))

# print('\n', best_cla_ams.get_params)
with open('learning_model_ams_dt.pkl', 'wb') as f:
    pickle.dump(dict(model=best_cla_ams), f)

# print(best_cla_acc.get_params)
with open('learning_model_acc.pkl', 'wb') as f:
    pickle.dump(dict(model=best_cla_acc), f)

winsound.Beep(300,500)