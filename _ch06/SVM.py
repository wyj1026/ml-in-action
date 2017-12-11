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
            fXi = float(multiply(alphas, label_matrix).T* \
                (data_matrix * data_matrix[i, :].T)) + b

            # Calculate error_rate
            Ei = fXi - float(label_matrix[i])

            # We hava to alphas if the result is far beyond the toler
            if ((label_matrix[i] * Ei < -toler) and
                (alphas[i] < C)) or ((label_matrix[i] * Ei > toler) and \
                (alphas[i] > 0)):

                # Choose the second variable
                j = select(i, m)

                print("%d, %d choosed\n" %(i, j))
                fXj = float(multiply(alphas, label_matrix).T * \
                    (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_matrix[j])

                # Store the old alphas
                alphaI_old = alphas[i].copy();
                alphaJ_old = alphas[j].copy();
                if (label_matrix[i] != label_matrix[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] -C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue

                # Page 128
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T \
                    - data_matrix[i, :] * data_matrix[i, :].T \
                    - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    continue

                # Update alphas
                alphas[j] -= label_matrix[j] * (Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJ_old) < 0.00001:
                    continue
                alphas[i] += label_matrix[j] * label_matrix[i] *\
                    (alphaJ_old - alphas[j])
                
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


class SMO:
    def __init__(self, data_matrix, label_matrix, C, toler):
        self.X = data_matrix
        self.label_matrix = label_matrix
        self.C = C
        self.tol = toler
        self.m = shape(data_matrix)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.e_cache = mat(zeros((self.m, 2)))


def calc_EK(ob, i):
    fXk = float(multiply(ob.alphas, ob.label_matrix).T * \
        (ob.X * ob.X[i,:].T)) + ob.b
    return fXk - float(ob.label_matrix[i])


def select_J(i, ob, Ei):
    max_K = -1
    max_delta_E = 0
    Ej = 0
    ob.e_cache[i] = [1, Ei]
    valid_E_cache = nonzero(ob.e_cache[:, 0].A)[0]
    if (len(valid_E_cache)) > 1:
        for k in valid_E_cache:
            if k == i:
                continue
            Ek = calc_EK(ob, k)
            delta_E = abs(Ei - Ek)
            if delta_E > max_delta_E:
                max_K = k;
                max_delta_E = delta_E
                Ej = Ek
        return max_K, Ej
    else:
        j = select(i, ob.m)
        Ej = calc_EK(ob, j)
    return j, Ej


def update_EK(ob, k):
    Ek = calc_EK(ob, k)
    ob.e_cache[k] = [1, Ek]


def inner_loop(i, ob):
    Ei = calc_EK(ob, i)
    if ((ob.label_matrix[i] * Ei < -ob.tol) and (ob.alphas[i] < ob.C)) or \
        ((ob.label_matrix[i] * Ei > ob.tol) and (ob.alphas[i] > 0 )):
        j, Ej = select_J(i, ob, Ei)
        alphaI_old = ob.alphas[i].copy()
        alphaJ_old = ob.alphas[j].copy()
        if (ob.label_matrix[i] != ob.label_matrix[j]):
            L = max(0, ob.alphas[j] - ob.alphas[i])
            H = min(ob.C, ob.C + ob.alphas[j] - ob.alphas[i])
        else:
            L = max(0, ob.alphas[j] + ob.alphas[i] - ob.C)
            H = min(ob.C, ob.alphas[j] + ob.alphas[i])
        if L == H:
            return 0
        eta = 2.0 * ob.X[i,:] * ob.X[j,:].T - ob.X[i, :] * ob.X[i, :].T - \
            ob.X[j, :] * ob.X[j, :].T
        if eta >= 0:
            return 0

        ob.alphas[j] -= ob.label_matrix[j] * (Ei - Ej)/eta
        ob.alphas[j] = clip_alpha(ob.alphas[j], H, L)
        update_EK(ob, j)
        if (abs(ob.alphas[j] - alphaJ_old) < 0.0001):
            return 0
        ob.alphas[i] += ob.label_matrix[j] * ob.label_matrix[i] * \
            (alphaJ_old - ob.alphas[j])
        update_EK(ob, i)

        b1 = ob.b - Ei - ob.label_matrix[i] * (ob.alphas[i] - alphaI_old) * \
            ob.X[i, :] * ob.X[i, :].T - ob.label_matrix[j] * \
            (ob.alphas[j] -alphaJ_old) * ob.X[i, :] * ob.X[j, :].T
        b2 = ob.b - Ej - ob.label_matrix[i] * (ob.alphas[i] - alphaI_old) * \
            ob.X[i, :] * ob.X[j, :].T - ob.label_matrix[j] * \
            (ob.alphas[j] -alphaJ_old) * ob.X[j, :] * ob.X[j, :].T
        if (0 < ob.alphas[i]) and (ob.C > ob.alphas[i]):
            ob.b = b1
        elif (0 < ob.alphas[j]) and (ob.C > ob.alphas[j]):
            ob.b = b2
        else:
            ob.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smo_platt(data_matrix, label_matrix, C, toler, max_iter, k_tup=('lin', 0)):
    ob = SMO(mat(data_matrix), mat(label_matrix).transpose(), C, toler)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0

    while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(ob.m):
                alpha_pairs_changed += inner_loop(i, ob)
                print("full set , iter :%d i:%d, pairs changed %d" % \
                        (iter, i, alpha_pairs_changed))
            iter += 1
        else:
            non_bound = nonzero((ob.alphas.A > 0) * (ob.alphas.A < C))[0]
            for i in non_bound:
                alpha_pairs_changed += inner_loop(i, ob)
                print("non-bound iter %d i:%d, pairs changed %d" %\
                            (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif (alpha_pairs_changed == 0):
            entire_set = True
    return ob.b, ob.alphas


def calculate_w(alphas, data, label):
    X = mat(data)
    L = mat(label).transpose()
    m, n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i] * L[i], X[i,:].T)
    return w


d, l = load_data('./Ch06/testSet.txt')
b, alphas = smo_platt(d, l, 0.6, 0.001, 40)
w = calculate_w(alphas, d, l)
print(b, w)


