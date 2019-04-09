def all_path(dataset):
    paths = list()
    if dataset == 'GH':
        paths.append('test_all.csv')
        paths.append('test_fork.csv')
        paths.append('test_watch.csv')
        paths.append('train_all.csv')
        paths.append('train_fork.csv')
        paths.append('train_watch.csv')
    elif dataset == 'SO':
        paths.append('test_all.csv')
        paths.append('test_answer.csv')
        paths.append('test_favorite.csv')
        paths.append('train_all.csv')
        paths.append('train_answer.csv')
        paths.append('train_favorite.csv')
    return paths


def extract_uid(path):
    file = open(path, 'r')
    uid = [l.split(',')[1] for l in file]
    return uid[1:]


def dict_uid(dataset):
    path = './data/full_data_v1/' + dataset + '_no_negative_train_test'
    uid = list()
    sub_paths = all_path(dataset=dataset)
    for p in sub_paths:
        uid += extract_uid(path=path + '/' + p)
    return list(set(uid))


if __name__ == '__main__':
    dataset = 'GH'
    uid = dict_uid(dataset=dataset)
    print(dataset, len(uid))

    dataset = 'SO'
    uid = dict_uid(dataset=dataset)
    print(dataset, len(uid))
