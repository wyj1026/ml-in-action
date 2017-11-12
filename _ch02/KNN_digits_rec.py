from numpy import *
from os import listdir
from KNN import classfy0

def img2vector(file):
    res = zeros((1,1024))
    with open(file) as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                res[0, 32*i+j] = int(line[j])
    return res


def handwriting_recognization():
    hw_labels = []
    training_list = listdir('./Ch02/trainingDigits/')
    training_matrix = zeros((len(training_list), 1024))
    for i in range(len(training_list)):
        file_name = training_list[i]
        file_str = file_name.split('.')[0]
        number = int(file_str.split('_')[0])
        hw_labels.append(number)
        training_matrix[i,:] = img2vector('./Ch02/trainingDigits/' + file_name)

    print(training_matrix.shape[0])
    test_list = listdir('./Ch02/testDigits')
    error_count = 0
    test_size = len(test_list)
    for i in range(test_size):
        file_name = test_list[i]
        file_str = file_name.split('.')[0]
        number = int(file_str.split('_')[0])
        vector2test = img2vector('./Ch02/testDigits/' + file_name)
        result = classfy0(vector2test, training_matrix, hw_labels, 3)
        # print('The classifier came back with %d, the real answer is %d' %(result, number))
        if number != result:
              error_count +=1
    print('The erro rate is %f' % (error_count/test_size))
#print(img2vector('./Ch,2/testDigits/0_13.txt'))

import time
start_time = time.time()
handwriting_recognization()
print("----%d seconds ----" %(time.time() - start_time))
