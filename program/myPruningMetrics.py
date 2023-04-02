import numpy as np
from sklearn.metrics import confusion_matrix

def mean_outs_of_single_weight(clf_reg, X, number, index):
    activation_i = X.copy()
    for i in range(number):
        sum_out = np.dot(activation_i, clf_reg.coefs_[i]) + clf_reg.intercepts_[i]
        if clf_reg.activ == "sigmoid":
            activation_i = clf_reg._sigmoid(sum_out)
        else:
            activation_i = clf_reg._relu(sum_out)
    return np.mean(activation_i[:,index[0]]*clf_reg.coefs_[number][index])

def outs_of_single_weight_for_variance(clf_reg, X):
    l_c = clf_reg.layers_count
    tmp_var = [None]*l_c
    tmp_mean = [None]*l_c

    activation_i = X.copy()
    for i in range(l_c):
        sh = clf_reg.coefs_[i].shape
        var_t = np.zeros(sh)
        mean_t = np.zeros(sh)        
        for j in range(sh[1]):
            tmp = activation_i*clf_reg.coefs_[i][:,j]
            var_t[:,j] = np.var(tmp, axis=0)
            mean_t[:,j] = np.mean(tmp, axis=0)
        tmp_var[i] = var_t.copy()
        tmp_mean[i] = mean_t.copy()

        if i == (l_c-1): #wartoœæ funkcji aktywacji na warstwie wyjœciowej nie jest potrzebna
            break

        sum_out = np.dot(activation_i, clf_reg.coefs_[i]) + clf_reg.intercepts_[i]
        if clf_reg.activ == "sigmoid":
            activation_i = clf_reg._sigmoid(sum_out)
        else:
            activation_i = clf_reg._relu(sum_out)
    return tmp_var, tmp_mean

def APER(y_true, y_pred): #miara potrzebna do kolejnej metody przycinania - tylko dla klasyfikacji; interpretowane jako 1-dok³adnoœæ
    matrix = confusion_matrix(y_true, y_pred)
    l_e = np.sum(matrix)
    return (l_e - np.trace(matrix))/l_e

def APERP(y_true, y_pred): #miara potrzebna do kolejnej metody przycinania - tylko dla klasyfikacji
    matrix = confusion_matrix(y_true, y_pred)
    sums = np.sum(matrix, axis=1)
    return np.sum((sums - np.diag(matrix))/sums)/matrix.shape[0]


def class_dE_zj(clf, x, y, layer, l_c): #chyba Ÿle rozumiem wzór - przycinanie Optimal Brain Damage
    activation = clf._forward(x)
    if layer == l_c-1:
        deri = activation[layer]*(1 - activation[layer]) #zmieniæ na pochodn¹ softmax, jakbym u¿y³ takiej funkci aktywacji
    elif clf.activ == "sigmoid":
        deri = activation[layer]*(1 - activation[layer])
    else:
        deri = (activation[layer]>0)*1
    return 0.5*deri*((-activation[layer])**2) #dla klasyfikacji w metodzie OBD - dopytaæ