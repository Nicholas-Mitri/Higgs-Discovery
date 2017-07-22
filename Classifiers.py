from collections import OrderedDict

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = {'dt': DecisionTreeClassifier(),
               'svm': SVC(),
               'bagging': BaggingClassifier(),
               'mlp': MLPClassifier(),
               'ada': AdaBoostClassifier()}

params = {'dt': OrderedDict([('criterion', ['entropy', 'gini']),
                             ('max_depth', [5, 10, 15]),
                             ('max_features', [0.2, 0.5, 0.8, 1]),
                             ('class_weight', [None, 'balanced'])]),
          'svm': OrderedDict([('C', [0.1, 0.5,1, 10]), ('gamma', [2**-5, 2**-3, 2**-2, 0.5, 1]), ('probability', [True])]),
          'bagging': OrderedDict([('criterion', ['entropy']), ('max_depth', [10]), ('n_estimators', [20])]),
          'ada': OrderedDict([('learning_rate', [1.0]), ('algorithm', ['SAMME.R']), ('n_estimators', [20])]),
          'mlp': OrderedDict([('activation', ['tanh', 'relu']),
                              ('alpha', [0.001, 0.1, 1, 10]),
                              ('hidden_layer_sizes', [(10, 10), (10, 20, 10), (50, 50, 50)]),
                              ('solver', ['lbfgs', 'sgd', 'adam'])])}

params_single = {'dt': OrderedDict([('criterion', ['entropy']),
                                    ('max_depth', [10]),
                                    ('max_features', [0.5]),
                                    ('class_weight', [None])]),
                 'svm': OrderedDict([('C', [10]), ('gamma', [0.125]), ('probability', [True])]),
                 'mlp': OrderedDict([('activation', ['tanh']),
                                     ('batch_size' , [2000]),
                                     ('alpha', [1]),
                                     ('hidden_layer_sizes', [(10,10)]),
                                     ('solver', ['lbfgs'])])}
