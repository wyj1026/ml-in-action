from numpy import *
def load_data(filename):
    data_matrix = []
    label_matrix = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line_array = line.strip().split('\t')
            data_matrix.append([float(line_array[0]),
                float(line_array[1])])
            label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix


def select(i, m):
    j = i
    while (j==i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    return min(h, max(l, aj))


# Input: data_matrix, label_matrix, C(the range of alpha), 
# error_tolrant rate and the max iteration number
def simple_SMO(data_matrix, label_matrix, C, toler, max_iter):
    data_matrix = mat(data_matrix)
    label_matrix = mat(label_matrix).transpose()
    b = 0;
    m, n = shape(data_matrix)
    alphas = mat(zeros((m,1)))

    # This interger stores the number of iteration that the value 
    # of alphas doesn't change
    iter = 0

    while (iter < max_iter):
        # Record whether we've got better alphas
        alpha_pairs_changed = 0

        # Traverse alphas
        for i in range(m):

            # Calculate g(xi) Page 129 KKK requirement
            # which also means the predict label of data_matrix[i]
            fXi = float(multiply(alphas, label_matrix).T* (data_matrix * data_matrix[i, :].T)) + b

            # Calculate error_rate
            Ei = fXi - float(label_matrix[i])
            print(Ei)

            # We hava to alphas if the result is far beyond the toler
            if ((label_matrix[i] * Ei < -1 * toler) and
                (alphas[i] < C)) or ((label_matrix[i] * Ei > toler) and (alphas[i] > 0)):

                # Choose the second variable
                j = select(i, m)

                print("%d, %d choosed\n" %(i, j))
                fXj = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_matrix[j])

                # Store the old alphas
                alphaI_old = alphas[i].copy();
                alphaJ_old = alphas[j].copy();
                if (label_matrix[i] != label_matrix[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[i] - alphas[j])
                else:
                    L = max(0, alphas[i] + alphas[j] -C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue

                # Page 128
                eta = 2 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    continue

                # Update alphas
                alphas[j] -= label_matrix[j] * (Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJ_old) < 0.00001:
                    continue
                alphas[i] += label_matrix[j] * label_matrix[i] * (alphaJ_old - alphas[j])
                
                # Update b
                b1 = b - Ei - label_matrix[i] * (alphas[i] - alphaI_old)*data_matrix[i, :]*data_matrix[i, :].T - label_matrix[j] * (alphas[j] - alphaJ_old) * data_matrix[i, :] *data_matrix[j,:].T
                b2 = b - Ej - label_matrix[i] * (alphas[i] - alphaI_old)*data_matrix[i, :]*data_matrix[j, :].T - label_matrix[j] * (alphas[j] - alphaJ_old) * data_matrix[j, :] *data_matrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d i :%d, pairs change %d" %(iter, i , alpha_pairs_changed))
                if (alpha_pairs_changed == 0):
                    iter += 1
                else:
                    iter = 0;
                print("iteration number: %d" % iter)
    return b, alphas


d, l = load_data('./Ch06/testSet.txt')
b, alphas = simple_SMO(d, l, 0.6, 0.001, 40)
print(b, alphas)
