def load_sample_data():
    data_matrix = ([
        [1. , 2.1],
        [2. , 1.1],
        [1.3, 1. ],
        [2., 1.]
    ])
    class_labels = [1, 1, -1, -1, 1]
    return data_matrix, class_labels

def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    ret_array = ones((shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[: , dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[: , dimen] > thresh_val] = -1.0
    return ret_array

def build_stump(data_array, class_labels, D):
    data_matrix = mat(data_array)
    label_matrix = mat(class_labels).T
    m, n = shape(data_matrix)
    num_steps = 10.0;
    best_stump = {}
    best_class_est = mat(zeros((m, 1)))
    min_error = inf
    for i in range(n):
        range_min = data_matrix[: , i].min()
        range_max = data_matrix[: , i].max()
        step_size = (range_max - range_min)/num_steps
        for j in range(-1, int(num_steps)+1):
            for inequal in ['lt','gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted = stump_classify(data_matrix, i, thresh_val, inequal)
                err_array = mat(ones((m, 1)))
                err_array[predicted == label_matrix] = 0
                weighted_error = D.T * err_array
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est

