from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
import numpy as np
from sklearn.metrics import average_precision_score


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
    for i in range(len(data)):
        splits = data[i].split(",")
        u_id = splits[0]
        if u_id in users_id:
            users_id[u_id].append(i)
        else:
            users_id[u_id] = [i]
    return users_id


def baseline_model(X_train, y_train, X_test, y_test, name):
    # vectorizer = TfidfVectorizer().fit(X_train)
    vectorizer = CountVectorizer().fit(X_train)
    X_train_trans = vectorizer.transform(X_train)
    X_test_trans = vectorizer.transform(X_test)
    y_train = np.array(y_train)

    if name == 'lr':
        model = LogisticRegression().fit(X=X_train_trans, y=y_train)
    elif name == 'svm':
        model = svm.SVC(kernel='linear', probability=True).fit(X=X_train_trans, y=y_train)
    elif name == 'tree':
        model = tree.DecisionTreeClassifier().fit(X=X_train_trans, y=y_train)

    pred = model.predict_proba(X_test_trans)[:, 1]
    return pred


def average_precision(u_id, labels, predicted):
    label, prediction = list(), list()
    for id in u_id:
        label.append(labels[id])
        prediction.append(predicted[id])
    return average_precision_score(y_true=label, y_score=prediction)


def MAP_users(users, labels, predicted):
    avgs = list()
    for k in users.keys():
        if len(users[k]) >= 10:
            avgs.append(average_precision(u_id=users[k], labels=labels, predicted=predicted))
    return sum(avgs) / len(avgs)


if __name__ == '__main__':
    train_path = './toy_data/training/user_answer_training.csv'
    # train_path = './toy_data/training/user_favorite_training.csv'
    # train_path = './toy_data/training/user_fork_training.csv'
    # train_path = './toy_data/training/user_watch_training.csv'
    train_features, train_lables = load_data(path=train_path)
    print(len(train_features), len(train_lables))

    test_path = './toy_data/test/user_answer_testing.csv'
    # test_path = './toy_data/test/user_favorite_testing.csv'
    # test_path = './toy_data/test/user_fork_testing.csv'
    # test_path = './toy_data/test/user_watch_testing.csv'
    test_features, test_lables = load_data(path=test_path)
    print(len(test_features), len(test_lables))

    test_users_repos = load_users_repos_data(path=test_path)
    # name = 'lr'
    # name = 'svm'
    name = 'tree'
    y_pred = baseline_model(X_train=train_features, y_train=train_lables, X_test=test_features, y_test=test_lables,
                            name=name)
    score = MAP_users(users=test_users_repos, labels=test_lables, predicted=y_pred)
    print(score)
