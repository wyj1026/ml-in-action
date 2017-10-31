from numpy import *
import KNN

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
        number =
        hw_labels[i] = number
        training_matrix[i,:] = img2vector(filename)

print(img2vector('./Ch02/testDigits/0_13.txt'))
