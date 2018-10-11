from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


def load_data(path):
    data = open(path, 'r').readlines()
    features, labels = list(), list()
    for d in data:
        splits = d.split(",")
        features.append(splits[2])
        labels.append(int(splits[3]))
    return features, labels


def load_users_repos_data(path):
    data = open(path, 'r').readlines()
    users_id = dict()
    for i in xrange(len(data)):
        splits = data[i].split(",")
        u_id = splits[0]
        if u_id in users_id:
            users_id[u_id].append(i)
        else:
            users_id[u_id] = [i]
    return users_id


def baseline_model(X_train, y_train, X_test, y_test, name):
    vectorizer = TfidfVectorizer().fit(X_train)
    X_train_trans = vectorizer.transform(X_train)
    X_test_trans = vectorizer.transform(X_test)
    y_train = np.array(y_train)
    print X_train_trans.shape, X_test_trans.shape
    print y_train.shape

    if name == 'lr':
        model = LogisticRegression().fit(X=X_train_trans, y=y_train)

    pred = model.predict_proba(X_test_trans)
    print pred.shape


if __name__ == '__main__':
    train_path = './toy_data/training/user_favorite_training.csv'
    train_features, train_lables = load_data(path=train_path)
    print len(train_features), len(train_lables)

    test_path = './toy_data/test/user_favorite_testing.csv'
    test_features, test_lables = load_data(path=test_path)
    print len(test_features), len(test_lables)

    test_users_repos = load_users_repos_data(path=test_path)

    name = 'lr'
    baseline_model(X_train=train_features, y_train=train_lables, X_test=test_features, y_test=test_lables, name=name)
