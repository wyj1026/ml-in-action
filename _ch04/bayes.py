from math import *
from numpy import *


def load_data_set():
    posting_list = [['my', ' dog', 'has', 'flea', 'problems',
                        'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog',
                        'park','stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I',
                        'love', 'him'],
                    ['stop', 'postiong', 'stupid', 'worthless',
                        'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                        'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog',
                        'food', 'stupid']]
    class_list = [0, 1, 0, 1, 0, 1]
    return posting_list, class_list



def create_vocabulary_list(data_set):
    vocab_set = set([])
    for d in data_set:
        vocab_set = vocab_set | set(d)
    return list(vocab_set)


# This function receive a word list and a doc, it returns
# a list of nums, presenting the numbers of the words that
# appear in the doc
def words2vector(vocab_list, input_set):
    vector = [0] * len(vocab_list)
    for word in vocab_list:
        if word in input_set:
            vector[vocab_list.index(word)] += 1
    return vector


def nb_train(matrix, category):
    num_of_docs = len(matrix)
    num_of_words = len(matrix[0])
    p_abusive = sum(category)/float(num_of_docs)

    p0_num = ones(num_of_words)
    p1_num = ones(num_of_words)
    p0_denom = 2.0;
    p1_denom = 2.0;

    for i in range(num_of_docs):
        if category[i] == 1:
            p1_num += matrix[i]
            p1_denom += sum(matrix[i])
        else:
            p0_num += matrix[i]
            p0_denom += sum(matrix[i])

    return log(exp(p0_num/p0_denom)), log(exp(p1_num/p1_denom)), p_abusive


def naive_bayes_classify(vector2classify, p0_vector, p1_vector, p_class):
    p1 = sum(vector2classify*p1_vector) + log(p_class);
    p0 = sum(vector2classify*p0_vector) + log(1 - p_class);
    if p1 > p0:
        return 1
    else:
        return 0


def naive_bayes_testing():
    train_cases, train_classes = load_data_set()
    vocabulary_list = create_vocabulary_list(train_cases)
    train_matrix = []
    for case in train_cases:
        train_matrix.append(words2vector(vocabulary_list, case))
    p0, p1, p_abusive = nb_train(train_matrix, train_classes)
    test_case1 = ['i', 'love', 'my', 'dog'] 
    test_case2 = ['you', 'are', 'stupid']

    test_matrix = [words2vector(vocabulary_list, test_case1),
                    words2vector(vocabulary_list, test_case2)]
    for i in test_matrix: 
        res =  naive_bayes_classify(i, p0, p1, p_abusive)
        print("Test case ", i, " classified as ", res) 
        
