from math import log
from imp import reload


# This function receives a data_set and returns its entropy, which
# describes the complex rate of the data, the more types of features the
# data hava, the bigger the shannon_entropy is.
def calc_shannon_entropy(data_set):
    num_of_data = len(data_set)
    label_counter = {}

    for vector in data_set:
        vector_label = vector[-1]
        if vector_label not in label_counter.keys():
            label_counter[vector_label] = 0
        label_counter[vector_label] += 1

    shannon_entropy = 0;
    for key in label_counter:
        prob = float(label_counter[key])/num_of_data
        shannon_entropy -= prob*log(prob, 2)
    return shannon_entropy


def split_data(data_set, pos, value):
    target_data_set = []
    for vector in data_set:
        if vector[pos] == value:
            reduced_vector = vector[:pos]
            reduced_vector.extend(vector[pos+1:])
            target_data_set.append(reduced_vector)
    return target_data_set


# This function receive a data_set and return the best feature
# to divide the input data_set.
# It calculate the base entropy at first and calculate new entropy
# with out a particular feature and update the best feature and best
# info_gain by calculating information gain
def choose_best_feature(data_set):
    features_num = len(data_set[0]) - 1
    entropy = calc_shannon_entropy(data_set)
    best_feature = -1
    best_info_gain = 0
    for i in range(features_num):
        feature_list = [example[i] for example in data_set]
        unique_feature = set(feature_list)
        new_entropy = 0
        for value in unique_feature:
            sub_data_set = split_data(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob*calc_shannon_entropy(sub_data_set)
        info_gain = entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def get_majority(l):
    class_count = {}
    for vote in l:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(),
        key=operator.itemgetter(1), reverse=Tre)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]

    # All instances belongs to the same calss.
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # There is no feature left.
    if len(data_set[0]) == 1:
        return get_majority(class_list)
    best_feature = choose_best_feature(data_set)
    best_feature_label = labels[best_feature]
    print("\nChoose %d feature, label: %s\n" %( best_feature, best_feature_label))
    my_tree = {best_feature_label: {}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in data_set]
    unique_feature_values = set(feature_values)
    for value in unique_feature_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data(data_set, best_feature, value), sub_labels)
        print(split_data(data_set, best_feature, value))
    return my_tree


def classify(tree, labels, test_vector):
    first_str = tree.keys()[0]
    second_dict = tree[first_str]
    feature_index = labels.index(first_str)
    for key in second_dict.keys():
        if test_vector[feature_index] == keys:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key],
                    labels, test_vector)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(tree, filename):
    import pickle
    with open(filename, 'w') as f:
        pickle.dump(tree, f)


def grab_tree(filename):
    import pickle
    with open(filename) as f:
        return pickle.load(f)
