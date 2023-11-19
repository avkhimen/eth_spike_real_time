import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import pickle

warnings.simplefilter('ignore')

offsets = [50,60,70,80,90,100]
nn_points = [17,19,21,23,25,27,39,31]

offsets = [90]
nn_points = [17]

for offset in offsets:
    for nn_point in nn_points:

        df = pd.read_csv('data/original_files/ETHXBT_60.csv', header=None,
                        names=['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2'])
        
        df = df.dropna()

        df['unix_timestamp'] = df['unix_timestamp'].astype(int)

        df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')

        df = df.drop(['unix_timestamp', 'other_1', 'other_2'], axis=1)

        df = df[['timestamp','open_price','high_price','low_price','close_price']]

        df = df.resample('4H', on='timestamp').agg({
            'open_price': 'first',
            'high_price': 'max',
            'low_price': 'min',
            'close_price': 'last'
            })

        df = df.dropna()

        df = df[['close_price']]

        df['close_price_perc'] = df['close_price'].pct_change()

        df = df[1:]

        df['close_price_perc'] = df['close_price_perc'] * 100

        threshold = 4 #originally 4

        df['spike'] = (df['close_price_perc'] > threshold).fillna(False)

        df = df[['close_price','spike']]

        offset = offset

        close_prices_list = sliding_window_view(df['close_price'], offset).tolist()

        spike_list = sliding_window_view(df['spike'], 1).tolist()[offset-1:]
        spike_list = [elem[0] for elem in spike_list]

        X = close_prices_list[:-1]
        y_ = spike_list[1:]

        y = []
        for elem in y_:
            y.append(str(elem))

        #print(df.head(10))
        #print(X[:5], len(X))
        #print(y[:5], len(y))

        oversample = SMOTE()
        rus = RandomUnderSampler(random_state=42)
        #X, y = oversample.fit_resample(X, y)

        X = np.array(X)

        X_ = []
        for arr in X:
            #X_.append(np.concatenate((arr, arr/min(arr), arr/max(arr), arr/arr[0], arr/arr[-1], [max(arr)], [min(arr)], [np.mean(arr)]), axis=0))
            X_.append(np.concatenate((arr, arr/min(arr), [min(arr)]), axis=0))

        X = np.array(X_).tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        #X_train, y_train = oversample.fit_resample(X_train, y_train)
        X_train, y_train = rus.fit_resample(X_train, y_train)

        #clf = svm.SVC()
        #clf = tree.DecisionTreeClassifier()
        #clf = HistGradientBoostingClassifier()

        names = [
            "Nearest Neighbors",
            # "Gaussian Process",
            # "Decision Tree",
            # "Random Forest",
            # "Neural Net",
            # "AdaBoost",
            # "Naive Bayes",
            # "QDA",
        ]

        classifiers = [
            KNeighborsClassifier(nn_point),
            # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            # DecisionTreeClassifier(max_depth=5, random_state=42),
            # RandomForestClassifier(
            #     max_depth=5, n_estimators=10, max_features=1, random_state=42
            # ),
            # MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            # AdaBoostClassifier(random_state=42),
            # GaussianNB(),
            # QuadraticDiscriminantAnalysis(),
        ]

        for name, clf in zip(names, classifiers):

            clf.fit(X_train, y_train)

            with open('model.pickle', 'wb') as f:
                pickle.dump(clf, f)

            yhat = clf.predict(X_test)

            acc = accuracy_score(y_test, yhat)
            #print(name, 'Accuracy: %.3f' % acc)
            #print('Count of true in training set', y_train.count('True'))
            #print('Count of true in test set', y_test.count('True'))

            total_count = 0
            count_wrong = 0
            count_num_true = 0
            count_true_neg = 0
            for i in range(len(y_test)):
                total_count += 1
                if yhat[i] != y_test[i]:
                    count_wrong += 1
                if y_test[i] == 'True':
                    count_num_true += 1
                    if yhat[i] != 'True':
                        count_true_neg += 1

            #print('total_count', total_count)
            #print('count_wrong', count_wrong)
            #print('count_true_neg', count_true_neg)
            #print('Real accuracy', (total_count - count_wrong)/total_count)
            #print('Count num true ratio', (count_num_true - count_true_neg)/count_num_true)
            print(nn_point, offset, 'Count num true ratio', (count_num_true - count_true_neg)/count_num_true)
            #print('---------------------------------------')