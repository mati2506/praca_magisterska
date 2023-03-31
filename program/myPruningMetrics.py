import numpy as np
from sklearn.metrics import confusion_matrix

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