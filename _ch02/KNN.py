from numpy import *
import operator


def filetomatrix(f):
    target_file = open(f)
    lines = target_file.readlines()
    number_of_lines = len(lines)
    matrix = zeros((number_of_lines, 3))
    index = 0
    vector = []

    for line in lines:
        line = line.strip()
        list_from_line = line.split('\t')
        matrix[index, :] = list_from_line[0:3]
        vector.append(int(list_from_line[-1]))
        index += 1
    return matrix, vector


def auto_norm(data):

    # We can get the min/max array from columns by using arg "0".
    min_values = data.min(0)
    max_values = data.max(0)
    ranges = max_values - min_values
    target_data_set = zeros(shape(data))
    size = data.shape[0]
    target_data_set = data - tile(min_values, (size, 1))
    target_data_set = target_data_set/tile(ranges, (size, 1))
    return target_data_set, ranges, min_values


def data_test():
    data_matrix, data_vector = filetomatrix("./Ch02/datingTestSet2.txt")
    data_matrix, ranges, min_values = auto_norm(data_matrix)
    error_count = 0
    for i in range(data_matrix.shape[0]):
        res = classfy0( data_matrix[i,:], data_matrix, data_vector, 3)
        if res != data_vector[i]:
            error_count += 1
    print("The total test size is %d, and the error rate is %f" \
        % (data_matrix.shape[0], error_count/data_matrix.shape[0]))


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# This method receives a test case, dataset, corresbonding labels, and the k.
def classfy0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    # tile returns inX for 1 time in row and dataSetSize time in column
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2;

    # Get the sum of **2 above
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    # Get the sorted position of distances.
    sorted_distance_arg = distances.argsort()
    classCount = {}

    for i in range(k):
        vote = labels[sorted_distance_arg[i]]
        classCount[vote] = classCount.get(vote, 0) + 1

    # The method operator.itemgetter(i) return a callable
    # object that fetches item form its operand.
    scC = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return scC[0][0]



data_test()
