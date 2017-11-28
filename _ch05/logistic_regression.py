from numpy import *


def load_data():
    data_matrix = []
    label_matrix = []
    with open('./Ch05/testSet.txt', 'r') as f:
        for line in f.readlines():
            line_array = line.strip().split()
            data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
            label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix


def sigmoid(x):
    return 1.0/(1 + exp(-x))


def grand_ascent(data_matrix, label_matrix):
    data_matrix = mat(data_matrix)
    label_matrix = mat(label_matrix).transpose()

    m, n = shape(data_matrix)
    alpha = 0.001
    cycles = 500
    weights = ones((n, 1))
    for k in range(cycles):
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stochasic_grand_ascent(data_matrix, label_matrix, num=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num):
        data_index = [x for x in range(m)]
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            random_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[random_index] * weights))
            error = label_matrix[random_index] - h
            improver = [feature*alpha*error for feature in data_matrix[random_index]]
            weights = weights + improver
            del(data_index[random_index])
    return weights


def classify_vector(int_x, weights):
    prob = sigmoid(sum(int_x*weights))
    return 1.0 if prob > 0.5 else 0.0


def colic_test():
    with open('./Ch05/horseColicTraining.txt', 'r') as f:
        training_set = []
        training_labels = []
        for l in f.readlines():
            splited_line = l.strip().split('\t')
            line_array = []
            for i in range(21):
                line_array.append(float(splited_line[i]))
            training_set.append(line_array)
            training_labels.append(float(splited_line[21]))
                
    training_weights = stochasic_grand_ascent(array(training_set), training_labels, 500)
    print(len(training_weights))
    error_counter = 0
    number_of_test_vector = 0
    with open('./Ch05/horseColicTest.txt', 'r') as f:
        for line in f.readlines():
            number_of_test_vector += 1
            current_line = line.strip().split('\t')
            line_array = []
            for i in range(21):
                line_array.append(float(current_line[i]))
            if int(classify_vector(array(line_array), training_weights)) != int(current_line[21]):
                error_counter += 1
    
    error_rate = float(error_counter/number_of_test_vector)
    print('The error rate is {0}!\n'.format(error_rate))

colic_test()
