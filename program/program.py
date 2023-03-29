import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score

class MLP_clf:
    def __init__(self, hidden=(10,10,10), epochs=250, eta=0.1, activation = "relu", shuffle=True, part=None, calc_s=True):
        self.hidden = [hidden] if type(hidden)==int else list(hidden)    #Liczba neuronów na kolejnych warstwach ukrytych
        self.layers_count = 2 if type(hidden)==int else len(hidden)+1 #Liczba warstw (ukrytych + wyjściowa)
        self.epochs = epochs    #Liczba epok
        self.eta = eta          #Współczynnik uczenia
        self.activ = activation #Funkcja aktywacji używana w warstwach ukrytych
        self.shuffle = shuffle  #Czy mieszać próbki w epokach
        self.part = part        #na jakiej czesci probek uczyć
        self.coefs_ = None      #wagi
        self.intercepts_ = None #biasy
        self.class_labels_ = None #nazwy klas
        self.class_type = None    #jakiego typu jest zmienna decyzyjna
        self.class_count = None   #liczba klas
        self.calc_s = calc_s      #czy obliczać parametr potrzebny w przycinaniu Karnina
        self.karnin_s = None      #parametr do metody Karnina
        
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _relu(self, x):
        return np.max([np.zeros(x.shape), x], axis=0)

    def _softmax(self, x):
        ex = np.exp(x)
        return ex/np.sum(ex)
    
    def _forward(self, X):
        activation = [None]*self.layers_count
        tmp = X.copy()
        for i in range(self.layers_count-1):
            sum_out = np.dot(tmp, self.coefs_[i]) + self.intercepts_[i]
            if self.activ == "sigmoid":
                tmp = self._sigmoid(sum_out)
            else:
                tmp = self._relu(sum_out)
            activation[i] = tmp.copy()
        activation[-1] = self._sigmoid(np.dot(tmp, self.coefs_[-1]) + self.intercepts_[-1]) #docelowo softmax, ale pochodna jest jakaś dziwna
        return activation
    
    def fit(self, X, Y, X_val=None, Y_val=None):
        samples, features = X.shape
        if self.class_labels_ is None:
            self.class_labels_ = np.unique(Y)
            self.class_type = self.class_labels_.dtype
            self.class_count = len(self.class_labels_)
        y = np.zeros((samples, self.class_count))
        for i in range(samples):
            y[i,np.where(self.class_labels_==Y[i])[0]] = 1

        if X_val is None or Y_val is None:
            X_val = X.copy()
            Y_val = Y.copy()
            
        self.coefs_ = [None]*self.layers_count
        self.intercepts_ = [None]*self.layers_count
        self.coefs_[0] = np.random.normal(size=(features,self.hidden[0]))
        self.intercepts_[0] = np.random.normal(size=self.hidden[0])
        for i in range(1,self.layers_count-1):
            self.coefs_[i] = np.random.normal(size=(self.hidden[i-1],self.hidden[i]))
            self.intercepts_[i] = np.random.normal(size=self.hidden[i])
        self.coefs_[-1] = np.random.normal(size=(self.hidden[-1],self.class_count))
        self.intercepts_[-1] = np.random.normal(size=self.class_count)

        if self.calc_s:
            init_coefs = copy.deepcopy(self.coefs_)

            self.karnin_s = [None]*self.layers_count
            for i in range(self.layers_count):
                self.karnin_s[i] = np.zeros(self.coefs_[i].shape)
        
        for epoch in range(self.epochs):
            if self.calc_s:
                last_coefs = copy.deepcopy(self.coefs_)

            ind = np.arange(samples)
            if self.shuffle:
                np.random.shuffle(ind)
            if self.part is not None:
                ind = ind[:int(samples*self.part)]
            
            for i in ind:
                activation = self._forward(X[i])
                deri = activation[-1]*(1 - activation[-1]) #docelowo pochodna softmax
                delta = (activation[-1] - y[i])*deri
                gradient = np.outer(activation[-2], delta)
                self.intercepts_[-1] -= self.eta*delta
                for j in range(self.layers_count-2,0,-1):
                    if self.activ == "sigmoid":
                        deri = activation[j]*(1 - activation[j])
                    else:
                        deri = (activation[j]>0)*1
                    delta = np.dot(delta, self.coefs_[j+1].T)*deri
                    self.coefs_[j+1] -= self.eta*gradient
                    gradient = np.outer(activation[j-1], delta)
                    self.intercepts_[j] -= self.eta*delta
                if self.activ == "sigmoid":
                    deri = activation[0]*(1 - activation[0])
                else:
                    deri = (activation[0]>0)*1
                delta = np.dot(delta, self.coefs_[1].T)*deri
                self.coefs_[1] -= self.eta*gradient
                gradient = np.outer(X[i], delta)
                self.intercepts_[0] -= self.eta*delta
                self.coefs_[0] -= self.eta*gradient

            if self.calc_s:
                for i in range(self.layers_count):
                    coef = self.coefs_[i]
                    if np.all(coef != init_coefs[i]):
                        self.karnin_s[i] += ((coef-last_coefs[i])**2)*(coef/(self.eta*(coef-init_coefs[i])))

            if accuracy_score(Y_val, self.predict(X_val)) == 1:
                print(f"Uczenie zakończone po {epoch+1} epokach. Osiągnięto dopasowanie.")
                break
    
    def predict(self, X):
        Y = np.zeros(X.shape[0], dtype=self.class_type)
        for i, w in enumerate(X):
            Y[i] = self.class_labels_[np.argmax(self._forward(w)[-1])]
        return Y
    
    def predict_proba(self, X):
        return np.array([self._forward(x)[-1] for x in X])

    def refit(self, X, Y, X_val, Y_val, ep):
        samples = X.shape[0]
        y = np.zeros((samples, self.class_count))
        for i in range(samples):
            y[i,np.where(self.class_labels_==Y[i])[0]] = 1

        zero_w = [None]*self.layers_count
        for i in range(self.layers_count):
            zero_w[i] = (self.coefs_[i] == 0)

        for _ in range(ep):
            ind = np.arange(samples)
            if self.shuffle:
                np.random.shuffle(ind)
            if self.part is not None:
                ind = ind[:int(samples*self.part)]

            for i in ind:
                activation = self._forward(X[i])
                deri = activation[-1]*(1 - activation[-1]) #docelowo pochodna softmax
                delta = (activation[-1] - y[i])*deri
                gradient = np.outer(activation[-2], delta)
                gradient[zero_w[-1]] = 0 #zachowanie zerowych wag z przycinania
                self.intercepts_[-1] -= self.eta*delta
                for j in range(self.layers_count-2,0,-1):
                    if self.activ == "sigmoid":
                        deri = activation[j]*(1 - activation[j])
                    else:
                        deri = (activation[j]>0)*1
                    delta = np.dot(delta, self.coefs_[j+1].T)*deri
                    self.coefs_[j+1] -= self.eta*gradient
                    gradient = np.outer(activation[j-1], delta)
                    gradient[zero_w[j]] = 0 #zachowanie zerowych wag z przycinania
                    self.intercepts_[j] -= self.eta*delta
                if self.activ == "sigmoid":
                    deri = activation[0]*(1 - activation[0])
                else:
                    deri = (activation[0]>0)*1
                delta = np.dot(delta, self.coefs_[1].T)*deri
                self.coefs_[1] -= self.eta*gradient
                gradient = np.outer(X[i], delta)
                gradient[zero_w[0]] = 0 #zachowanie zerowych wag z przycinania
                self.intercepts_[0] -= self.eta*delta
                self.coefs_[0] -= self.eta*gradient

            if accuracy_score(Y_val, self.predict(X_val)) == 1:
                break
    
    def get_number_of_parametrs(self):
        res = 0
        for i in range(self.layers_count):
            w, h = self.coefs_[i].shape
            res += (self.intercepts_[i].shape[0] + w*h)
        return res

    def get_nuber_of_coefs(self):
        num = 0
        for coef in self.coefs_:
            w, h = coef.shape
            num += w*h
        return num

    def outs_of_single_weight(self, X, number, index):
        w = X.shape[0]
        outs = np.zeros(w)
        for j in range(w):
            activation_i = X[j].copy()
            for i in range(number):
                sum_out = np.dot(activation_i, self.coefs_[i]) + self.intercepts_[i]
                activation_i = self._sigmoid(sum_out)
            outs[j] = activation_i[index[0]]*self.coefs_[number][index]
        return outs



class MLP_reg:
    def __init__(self, hidden=(10,10,10), epochs=250, eta=0.1, activation = "relu", shuffle=True, part=None, calc_s=True):
        self.hidden = [hidden] if type(hidden)==int else list(hidden)    #Liczba neuronów na kolejnych warstwach ukrytych
        self.layers_count = 2 if type(hidden)==int else len(hidden)+1 #Liczba warstw (ukrytych + wyjściowa)
        self.epochs = epochs    #Liczba epok
        self.eta = eta          #Współczynnik uczenia
        self.activ = activation #Funkcja aktywacji używana w warstwach ukrytych
        self.shuffle = shuffle  #Czy mieszać próbki w epokach
        self.part = part        #na jakiej czesci probek uczyć
        self.coefs_ = None      #wagi
        self.intercepts_ = None #biasy
        self.calc_s = calc_s      #czy obliczać parametr potrzebny w przycinaniu Karnina
        self.karnin_s = None      #parametr do metody Karnina
        
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _relu(self, x):
        return np.max([np.zeros(x.shape), x], axis=0)

    def _identity(self, x):
        return x
    
    def _forward(self, X):
        activation = [None]*self.layers_count
        tmp = X.copy()
        for i in range(self.layers_count-1):
            sum_out = np.dot(tmp, self.coefs_[i]) + self.intercepts_[i]
            if self.activ == "sigmoid":
                tmp = self._sigmoid(sum_out)
            else:
                tmp = self._relu(sum_out)
            activation[i] = tmp.copy()
        activation[-1] = self._identity(np.dot(tmp, self.coefs_[-1]) + self.intercepts_[-1])
        return activation
    
    def fit(self, X, Y, X_val=None, Y_val=None):
        samples, features = X.shape

        if X_val is None or Y_val is None:
            X_val = X.copy()
            Y_val = Y.copy()
            
        self.coefs_ = [None]*self.layers_count
        self.intercepts_ = [None]*self.layers_count
        self.coefs_[0] = np.random.normal(size=(features,self.hidden[0]))
        self.intercepts_[0] = np.random.normal(size=self.hidden[0])
        for i in range(1,self.layers_count-1):
            self.coefs_[i] = np.random.normal(size=(self.hidden[i-1],self.hidden[i]))
            self.intercepts_[i] = np.random.normal(size=self.hidden[i])
        self.coefs_[-1] = np.random.normal(size=(self.hidden[-1],1))
        self.intercepts_[-1] = np.random.normal(size=1)

        if self.calc_s:
            init_coefs = copy.deepcopy(self.coefs_)

            self.karnin_s = [None]*self.layers_count
            for i in range(self.layers_count):
                self.karnin_s[i] = np.zeros(self.coefs_[i].shape)
        
        for epoch in range(self.epochs):
            if self.calc_s:
                last_coefs = copy.deepcopy(self.coefs_)

            ind = np.arange(samples)
            if self.shuffle:
                np.random.shuffle(ind)
            if self.part is not None:
                ind = ind[:int(samples*self.part)]
            
            for i in ind:
                activation = self._forward(X[i])
                deri = 1 #pochodna x to 1 (funkcja aktywacji jest funkcją liniową)
                delta = (activation[-1] - Y[i])*deri
                gradient = np.outer(activation[-2], delta)
                self.intercepts_[-1] -= self.eta*delta
                for j in range(self.layers_count-2,0,-1):
                    if self.activ == "sigmoid":
                        deri = activation[j]*(1 - activation[j])
                    else:
                        deri = (activation[j]>0)*1
                    delta = np.dot(delta, self.coefs_[j+1].T)*deri
                    self.coefs_[j+1] -= self.eta*gradient
                    gradient = np.outer(activation[j-1], delta)
                    self.intercepts_[j] -= self.eta*delta
                if self.activ == "sigmoid":
                    deri = activation[0]*(1 - activation[0])
                else:
                    deri = (activation[0]>0)*1
                delta = np.dot(delta, self.coefs_[1].T)*deri
                self.coefs_[1] -= self.eta*gradient
                gradient = np.outer(X[i], delta)
                self.intercepts_[0] -= self.eta*delta
                self.coefs_[0] -= self.eta*gradient

            if self.calc_s:
                for i in range(self.layers_count):
                    coef = self.coefs_[i]
                    if np.all(coef != init_coefs[i]):
                        self.karnin_s[i] += ((coef-last_coefs[i])**2)*(coef/(self.eta*(coef-init_coefs[i])))

            if mean_squared_error(Y_val, self.predict(X_val)) <= 1e-5:
                print(f"Uczenie zakończone po {epoch+1} epokach. Osiągnięto zakładny błąd.")
                break
    
    def predict(self, X):
        return np.array([self._forward(x)[-1] for x in X])

    def refit(self, X, Y, X_val, Y_val, ep):
        samples = X.shape[0]
        zero_w = [None]*self.layers_count
        for i in range(self.layers_count):
            zero_w[i] = (self.coefs_[i] == 0)

        for _ in range(ep):
            ind = np.arange(samples)
            if self.shuffle:
                np.random.shuffle(ind)
            if self.part is not None:
                ind = ind[:int(samples*self.part)]
            
            for i in ind:
                activation = self._forward(X[i])
                deri = 1 #pochodna x to 1 (funkcja aktywacji jest funkcją liniową)
                delta = (activation[-1] - Y[i])*deri
                gradient = np.outer(activation[-2], delta)
                gradient[zero_w[-1]] = 0 #zachowanie zerowych wag z przycinania
                self.intercepts_[-1] -= self.eta*delta
                for j in range(self.layers_count-2,0,-1):
                    if self.activ == "sigmoid":
                        deri = activation[j]*(1 - activation[j])
                    else:
                        deri = (activation[j]>0)*1
                    delta = np.dot(delta, self.coefs_[j+1].T)*deri
                    self.coefs_[j+1] -= self.eta*gradient
                    gradient = np.outer(activation[j-1], delta)
                    gradient[zero_w[j]] = 0 #zachowanie zerowych wag z przycinania
                    self.intercepts_[j] -= self.eta*delta
                if self.activ == "sigmoid":
                    deri = activation[0]*(1 - activation[0])
                else:
                    deri = (activation[0]>0)*1
                delta = np.dot(delta, self.coefs_[1].T)*deri
                self.coefs_[1] -= self.eta*gradient
                gradient = np.outer(X[i], delta)
                gradient[zero_w[0]] = 0 #zachowanie zerowych wag z przycinania
                self.intercepts_[0] -= self.eta*delta
                self.coefs_[0] -= self.eta*gradient

            if mean_squared_error(Y_val, self.predict(X_val)) <= 1e-5:
                break
    
    def get_number_of_parametrs(self):
        res = 0
        for i in range(self.layers_count):
            w, h = self.coefs_[i].shape
            res += (self.intercepts_[i].shape[0] + w*h)
        return res

    def get_nuber_of_coefs(self):
        num = 0
        for coef in self.coefs_:
            w, h = coef.shape
            num += w*h
        return num

    def outs_of_single_weight(self, X, number, index):
        w = X.shape[0]
        outs = np.zeros(w)
        for j in range(w):
            activation_i = X[j].copy()
            for i in range(number):
                sum_out = np.dot(activation_i, self.coefs_[i]) + self.intercepts_[i]
                activation_i = self._sigmoid(sum_out)
            outs[j] = activation_i[index[0]]*self.coefs_[number][index]
        return outs



def simple_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_coefs = clf_reg.get_nuber_of_coefs()
    del_w = 0 #liczba usuniętych wag
    del_n = 0 #liczba usuniętych neuronów

    tmp_w = copy.deepcopy(clf_reg.coefs_)

    for i in range(l_c):
        tmp_w[i][tmp_w[i] == 0] = np.nan
        del_w += np.sum(tmp_w[i] == 0)

    tmp_ind = [None]*l_c
    tmp_val = [None]*l_c
    while del_w < num_of_coefs:      
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)
        
        for i in range(l_c):
            if np.sum(np.isnan(tmp_w[i])) == np.size(tmp_w[i]):
                tmp_ind[i] = (0,0)
                tmp_val[i] = np.nan
            else:
                tmp_ind[i] = np.unravel_index(np.nanargmin(np.abs(tmp_w[i])),shape=tmp_w[i].shape)
                tmp_val[i] = tmp_w[i][tmp_ind[i]]
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        tmp_w[tmp][tmp_ind[tmp]] = np.nan
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if tmp_w[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(tmp_w[i], axis=1)
                    for ind in np.where(sums==0)[0]:
                        tmp_w[i] = np.delete(tmp_w[i], ind, 0)
                        tmp_w[i-1] = np.delete(tmp_w[i-1], ind, 1)
                        del_w += np.sum(clf_reg.coefs_[i-1][:,ind] != 0) #usunięte wagi z kolumny warstwy poprzedzającej (potrzebne do zakończenia głównego while)
                        del_n += 1
                        clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                        clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                        clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi, może gdzieś tą wartośc by dodawać???
                        if tmp_w[i].shape[0] < 2:
                            break

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        del_w += 1
    for i in range(l_c-1): #aktualizacja liczby neuronów w warstwach ukrytych po przycinaniu
        clf_reg.hidden[i] = clf_reg.coefs_[i].shape[1]
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar

def simple_pruning_amendment(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_coefs = clf_reg.get_nuber_of_coefs()
    del_w = 0 #liczba usuniętych wag
    del_n = 0 #liczba usuniętych neuronów

    tmp_w = copy.deepcopy(clf_reg.coefs_)
    c_r_clc = copy.deepcopy(clf_reg)

    for i in range(l_c):
        tmp_w[i][tmp_w[i] == 0] = np.nan
        del_w += np.sum(tmp_w[i] == 0)

    tmp_ind = [None]*l_c
    tmp_val = [None]*l_c
    del_n_n = [np.array([])]*(l_c+1)
    while del_w < num_of_coefs:      
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)
        
        for i in range(l_c):
            if np.sum(np.isnan(tmp_w[i])) == np.size(tmp_w[i]):
                tmp_ind[i] = (0,0)
                tmp_val[i] = np.nan
            else:
                tmp_ind[i] = np.unravel_index(np.nanargmin(np.abs(tmp_w[i])),shape=tmp_w[i].shape)
                tmp_val[i] = tmp_w[i][tmp_ind[i]]
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        tmp_w[tmp][tmp_ind[tmp]] = np.nan
        ind_w = list(tmp_ind[tmp])
        ind_w = (ind_w[0]+np.sum(del_n_n[tmp] <= ind_w[0]), ind_w[1]+np.sum(del_n_n[tmp+1] <= ind_w[1]))
        clf_reg.intercepts_[tmp][tmp_ind[tmp][1]] += np.mean(c_r_clc.outs_of_single_weight(X_t, tmp, ind_w))
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if tmp_w[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(tmp_w[i], axis=1)
                    for ind in np.where(sums==0)[0]:
                        del_n_n[i][del_n_n[i] > ind] -= 1 #przesunięcie, aby uniknąć pominięcia usuniętego wiersza, gdy najpierw usuwany jest wiersz o wyższym indeksie, a później o niższym
                        del_n_n[i] = np.append(del_n_n[i], [ind])
                        tmp_w[i] = np.delete(tmp_w[i], ind, 0)
                        tmp_w[i-1] = np.delete(tmp_w[i-1], ind, 1)
                        del_w += np.sum(clf_reg.coefs_[i-1][:,ind] != 0) #usunięte wagi z kolumny warstwy poprzedzającej (potrzebne do zakończenia głównego while)
                        del_n += 1
                        clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                        clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                        clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi, może gdzieś tą wartośc by dodawać???
                        if tmp_w[i].shape[0] < 2:
                            break

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        del_w += 1
    for i in range(l_c-1): #aktualizacja liczby neuronów w warstwach ukrytych po przycinaniu
        clf_reg.hidden[i] = clf_reg.coefs_[i].shape[1]
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar

def karnin_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_coefs = clf_reg.get_nuber_of_coefs()
    del_w = 0 #liczba usuniętych wag
    del_n = 0 #liczba usuniętych neuronów

    s = copy.deepcopy(clf_reg.karnin_s)
    c_r_clc = copy.deepcopy(clf_reg)

    tmp_ind = [None]*l_c
    tmp_val = [None]*l_c
    del_n_n = [np.array([])]*(l_c+1)
    while del_w < num_of_coefs:      
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)
        
        for i in range(l_c):
            if np.sum(np.isnan(s[i])) == np.size(s[i]):
                tmp_ind[i] = (0,0)
                tmp_val[i] = np.nan
            else:
                tmp_ind[i] = np.unravel_index(np.nanargmin(np.abs(s[i])),shape=s[i].shape)
                tmp_val[i] = s[i][tmp_ind[i]]
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        s[tmp][tmp_ind[tmp]] = np.nan
        ind_w = list(tmp_ind[tmp])
        ind_w = (ind_w[0]+np.sum(del_n_n[tmp] <= ind_w[0]), ind_w[1]+np.sum(del_n_n[tmp+1] <= ind_w[1]))
        clf_reg.intercepts_[tmp][tmp_ind[tmp][1]] += np.mean(c_r_clc.outs_of_single_weight(X_t, tmp, ind_w))
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if s[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(s[i], axis=1)
                    for ind in np.where(sums==0)[0]:
                        del_n_n[i][del_n_n[i] > ind] -= 1 #przesunięcie, aby uniknąć pominięcia usuniętego wiersza, gdy najpierw usuwany jest wiersz o wyższym indeksie, a później o niższym
                        del_n_n[i] = np.append(del_n_n[i], [ind])
                        s[i] = np.delete(s[i], ind, 0)
                        s[i-1] = np.delete(s[i-1], ind, 1)
                        del_w += np.sum(clf_reg.coefs_[i-1][:,ind] != 0) #usunięte wagi z kolumny warstwy poprzedzającej (potrzebne do zakończenia głównego while)
                        del_n += 1
                        clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                        clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                        clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi, może gdzieś tą wartośc by dodawać???
                        if s[i].shape[0] < 2:
                            break

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        del_w += 1
    for i in range(l_c-1): #aktualizacja liczby neuronów w warstwach ukrytych po przycinaniu
        clf_reg.hidden[i] = clf_reg.coefs_[i].shape[1]
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar

def pruning_by_variance(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_coefs = clf_reg.get_nuber_of_coefs()
    del_w = 0 #liczba usuniętych wag
    del_n = 0 #liczba usuniętych neuronów

    tmp_var = [None]*l_c
    tmp_mean = [None]*l_c
    for i in range(l_c):
        sh = clf_reg.coefs_[i].shape
        var_t = np.zeros(sh)
        mean_t = np.zeros(sh)
        for j in range(sh[0]):
            for k in range(sh[1]):
                outs = clf_reg.outs_of_single_weight(X_t, i, (j,k))
                var_t[j,k] = np.var(outs)
                mean_t[j,k] = np.mean(outs)
        tmp_var[i] = var_t.copy()
        tmp_mean[i] = mean_t.copy()

    tmp_ind = [None]*l_c
    tmp_val = [None]*l_c
    while del_w < num_of_coefs:      
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)
        
        for i in range(l_c):
            if np.sum(np.isnan(tmp_var[i])) == np.size(tmp_var[i]):
                tmp_ind[i] = (0,0)
                tmp_val[i] = np.nan
            else:
                tmp_ind[i] = np.unravel_index(np.nanargmin(np.abs(tmp_var[i])),shape=tmp_var[i].shape)
                tmp_val[i] = tmp_var[i][tmp_ind[i]]
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        tmp_var[tmp][tmp_ind[tmp]] = np.nan
        clf_reg.intercepts_[tmp][tmp_ind[tmp][1]] += tmp_mean[tmp][tmp_ind[tmp]]
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if tmp_var[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(tmp_var[i], axis=1)
                    for ind in np.where(sums==0)[0]:
                        tmp_var[i] = np.delete(tmp_var[i], ind, 0)
                        tmp_var[i-1] = np.delete(tmp_var[i-1], ind, 1)
                        tmp_mean[i] = np.delete(tmp_mean[i], ind, 0)
                        tmp_mean[i-1] = np.delete(tmp_mean[i-1], ind, 1)
                        del_w += np.sum(clf_reg.coefs_[i-1][:,ind] != 0) #usunięte wagi z kolumny warstwy poprzedzającej (potrzebne do zakończenia głównego while)
                        del_n += 1
                        clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                        clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                        clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi, może gdzieś tą wartośc by dodawać???
                        if tmp_var[i].shape[0] < 2:
                            break

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        del_w += 1
    for i in range(l_c-1): #aktualizacja liczby neuronów w warstwach ukrytych po przycinaniu
        clf_reg.hidden[i] = clf_reg.coefs_[i].shape[1]
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar


def FBI_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_hidden_neurons = np.sum(clf_reg.hidden)
    del_n = 0

    tmp_ind = [None]*l_c #numer neuronu z każdej warstwy, który jest kandydatem do usunięcia
    tmp_val = [None]*l_c #wartość zmiannej decyzyjnej dla tego neuronu
    tmp_ind[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    tmp_val[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    while del_n < (num_of_hidden_neurons - (l_c-1)): #odjęta liczba warstw ukrytych, bo w każdej warstwie musi zostać conajmiej 1 neuron
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)
        
        if if_clf:
            y = np.zeros((y_t.shape[0], clf_reg.class_count)) #tablica potrzebna do obliczenia błędów średniokwadratowych w przypadku klasyfikacji
            for i in range(y_t.shape[0]):
                y[i,np.where(clf_reg.class_labels_==y_t[i])[0]] = 1
        else:
            y = y_t.copy()
        for i in range(1,l_c):
            n_n_i_l = clf_reg.coefs_[i].shape[0] #liczba neuroów w danej warstwie ukrytej
            if n_n_i_l < 2:
                tmp_ind[i] = 0
                tmp_val[i] = np.nan
            else:
                Sj = np.zeros(n_n_i_l)
                for j in range(n_n_i_l):
                    tmp_net = copy.deepcopy(clf_reg)
                    tmp_net.coefs_[i][j,:] = 0 #ustawienie wag wyjściowych z neuronu na 0 - zasymulowanie, że wartość neuronu jest zerowa
                    if if_clf:
                        y_pred = tmp_net.predict_proba(X_t)
                    else:
                        y_pred = tmp_net.predict(X_t)
                    Sj[j] = mean_squared_error(y, y_pred)
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #numer (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #numer neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        clf_reg.hidden[tmp-1] -= 1 #aktualizacja liczby neuronów w warstwie ukrytej, z której nauron jest usuwany
        del_n += 1
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar

def APER(y_true, y_pred): #miara potrzebna do kolejnej metody przycinania - tylko dla klasyfikacji; interpretowane jako 1-dokładność
    matrix = confusion_matrix(y_true, y_pred)
    l_e = np.sum(matrix)
    return (l_e - np.trace(matrix))/l_e

def APERT_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_hidden_neurons = np.sum(clf_reg.hidden)
    del_n = 0

    tmp_ind = [None]*l_c #numer neuronu z każdej warstwy, który jest kandydatem do usunięcia
    tmp_val = [None]*l_c #wartość zmiannej decyzyjnej dla tego neuronu
    tmp_ind[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    tmp_val[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    while del_n < (num_of_hidden_neurons - (l_c-1)): #odjęta liczba warstw ukrytych, bo w każdej warstwie musi zostać conajmiej 1 neuron
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)

        y_pred0 = clf_reg.predict(X_t) #predykcja przed daną itracją przycinania
        for i in range(1,l_c):
            n_n_i_l = clf_reg.coefs_[i].shape[0] #liczba neuroów w danej warstwie ukrytej
            if n_n_i_l < 2:
                tmp_ind[i] = 0
                tmp_val[i] = np.nan
            else:
                Sj = np.zeros(n_n_i_l)
                for j in range(n_n_i_l):
                    tmp_net = copy.deepcopy(clf_reg)
                    tmp_net.coefs_[i][j,:] = 0 #ustawienie wag wyjściowych z neuronu na 0 - zasymulowanie, że wartość neuronu jest zerowa
                    y_pred = tmp_net.predict(X_t)
                    if if_clf:
                        Sj[j] = APER(y_t, y_pred) - APER(y_t, y_pred0)
                    else:
                        Sj[j] = mean_squared_error(y_t, y_pred) - mean_squared_error(y_t, y_pred0) #dla regresji miara APER zastępiona błędem średniokwadratowym
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #numer (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #numer neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        clf_reg.hidden[tmp-1] -= 1 #aktualizacja liczby neuronów w warstwie ukrytej, z której nauron jest usuwany
        del_n += 1
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar

def APERP(y_true, y_pred): #miara potrzebna do kolejnej metody przycinania - tylko dla klasyfikacji
    matrix = confusion_matrix(y_true, y_pred)
    sums = np.sum(matrix, axis=1)
    return np.sum((sums - np.diag(matrix))/sums)/matrix.shape[0]

def APERTP_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania; dla regresji działa identycznie, jak APERT
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_hidden_neurons = np.sum(clf_reg.hidden)
    del_n = 0

    tmp_ind = [None]*l_c #numer neuronu z każdej warstwy, który jest kandydatem do usunięcia
    tmp_val = [None]*l_c #wartość zmiannej decyzyjnej dla tego neuronu
    tmp_ind[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    tmp_val[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    while del_n < (num_of_hidden_neurons - (l_c-1)): #odjęta liczba warstw ukrytych, bo w każdej warstwie musi zostać conajmiej 1 neuron
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)

        y_pred0 = clf_reg.predict(X_t) #predykcja przed daną itracją przycinania
        for i in range(1,l_c):
            n_n_i_l = clf_reg.coefs_[i].shape[0] #liczba neuroów w danej warstwie ukrytej
            if n_n_i_l < 2:
                tmp_ind[i] = 0
                tmp_val[i] = np.nan
            else:
                Sj = np.zeros(n_n_i_l)
                for j in range(n_n_i_l):
                    tmp_net = copy.deepcopy(clf_reg)
                    tmp_net.coefs_[i][j,:] = 0 #ustawienie wag wyjściowych z neuronu na 0 - zasymulowanie, że wartość neuronu jest zerowa
                    y_pred = tmp_net.predict(X_t)
                    if if_clf:
                        Sj[j] = APERP(y_t, y_pred) - APERP(y_t, y_pred0)
                    else:
                        Sj[j] = mean_squared_error(y_t, y_pred) - mean_squared_error(y_t, y_pred0) #dla regresji miara APERP zastępiona błędem średniokwadratowym
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #numer (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #numer neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        clf_reg.hidden[tmp-1] -= 1 #aktualizacja liczby neuronów w warstwie ukrytej, z której nauron jest usuwany
        del_n += 1
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar

def PD_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_hidden_neurons = np.sum(clf_reg.hidden)
    del_n = 0

    tmp_ind = [None]*l_c #numer neuronu z każdej warstwy, który jest kandydatem do usunięcia
    tmp_val = [None]*l_c #wartość zmiannej decyzyjnej dla tego neuronu
    tmp_ind[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    tmp_val[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    while del_n < (num_of_hidden_neurons - (l_c-1)): #odjęta liczba warstw ukrytych, bo w każdej warstwie musi zostać conajmiej 1 neuron
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)

        activ = clf_reg._forward(X_t)
        for i in range(1,l_c):
            n_n_i_l = clf_reg.coefs_[i].shape[0] #liczba neuroów w danej warstwie ukrytej
            if n_n_i_l < 2:
                tmp_ind[i] = 0
                tmp_val[i] = np.nan
            else:
                Sj = np.mean(np.sum(clf_reg.coefs_[i-1]**2, axis=0)*(activ[i-1]**2), axis=0)
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #numer (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #numer neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        clf_reg.hidden[tmp-1] -= 1 #aktualizacja liczby neuronów w warstwie ukrytej, z której nauron jest usuwany
        del_n += 1
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar

def PEB_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_hidden_neurons = np.sum(clf_reg.hidden)
    del_n = 0

    tmp_ind = [None]*l_c #numer neuronu z każdej warstwy, który jest kandydatem do usunięcia
    tmp_val = [None]*l_c #wartość zmiannej decyzyjnej dla tego neuronu
    tmp_ind[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    tmp_val[0] = np.nan #neurony wejściowe (atrybuty) nie są przycinane; pominięcie warstwy wejściowej
    while del_n < (num_of_hidden_neurons - (l_c-1)): #odjęta liczba warstw ukrytych, bo w każdej warstwie musi zostać conajmiej 1 neuron
        last_w = copy.deepcopy(clf_reg.coefs_)
        last_b = copy.deepcopy(clf_reg.intercepts_)

        activ = clf_reg._forward(X_t)
        for i in range(1,l_c):
            n_n_i_l = clf_reg.coefs_[i].shape[0] #liczba neuroów w danej warstwie ukrytej
            if n_n_i_l < 2:
                tmp_ind[i] = 0
                tmp_val[i] = np.nan
            else:
                Sj = np.mean(np.sum(clf_reg.coefs_[i-1], axis=0)*(activ[i-1]), axis=0)
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #numer (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #numer neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = accuracy_score(y_v, clf_reg.predict(X_v))
            if acc > in_acc: #dokładność wzrosła, od teraz maksymalna utrata dokłądności liczona względem wyższej dokładności
                in_acc = acc
            elif acc < in_acc*(1-lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        else:
            error = mean_squared_error(y_v, clf_reg.predict(X_v))
            if error < in_error: #błąd zmalał, od teraz maksymalny wzrost błędu liczony względem mniejszego błędu
                in_error = error
            elif error > in_error*(1+lost):
                clf_reg.coefs_ = copy.deepcopy(last_w)
                clf_reg.intercepts_ = copy.deepcopy(last_b)
                break
        clf_reg.hidden[tmp-1] -= 1 #aktualizacja liczby neuronów w warstwie ukrytej, z której nauron jest usuwany
        del_n += 1
    if if_clf:
        miar = accuracy_score(y_v, clf_reg.predict(X_v))
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar




def class_dE_zj(clf, x, y, layer, l_c): #chyba źle rozumiem wzór - przycinanie Optimal Brain Damage
    activation = clf._forward(x)
    if layer == l_c-1:
        deri = activation[layer]*(1 - activation[layer]) #zmienić na pochodną softmax, jakbym użył takiej funkci aktywacji
    elif clf.activ == "sigmoid":
        deri = activation[layer]*(1 - activation[layer])
    else:
        deri = (activation[layer]>0)*1
    return 0.5*deri*((-activation[layer])**2) #dla klasyfikacji w metodzie OBD - dopytać

   

#test działania
data = pd.read_csv("iris.data")
X = data.iloc[:,0:4].values
Y = data.iloc[:,4].values
X_train, X_test, y_train, y_test = train_test_split(X, Y)
clf = MLP_clf(epochs=100, activation="sigmoid")
clf.fit(X_train, y_train)
#print(clf.coefs_)
print(accuracy_score(y_train, clf.predict(X_train)))

ll = 0.05

#clf1 = copy.deepcopy(clf)
#a, d1 = simple_pruning(clf1, ll, X_train, y_train)
#print(a)
#print(d1)
##print(clf1.coefs_)
#print(clf1.hidden)

#print(accuracy_score(y_train, clf1.predict(X_train)))
#print(accuracy_score(y_test, clf1.predict(X_test)))
#print()


#clf2 = copy.deepcopy(clf)
#b, d2 = simple_pruning_amendment(clf2, ll, X_train, y_train)
#print(b)
#print(d2)
##print(clf2.coefs_)
#print(clf2.hidden)

#print(accuracy_score(y_train, clf2.predict(X_train)))
#print(accuracy_score(y_test, clf2.predict(X_test)))
#print()


#clf3 = copy.deepcopy(clf)
#c, d3 = karnin_pruning(clf3, ll, X_train, y_train)
#print(c)
#print(d3)
##print(clf3.coefs_)
#print(clf3.hidden)

#print(accuracy_score(y_train, clf3.predict(X_train)))
#print(accuracy_score(y_test, clf3.predict(X_test)))
#print()


#clf4 = copy.deepcopy(clf)
#d, d4 = pruning_by_variance(clf4, ll, X_train, y_train)
#print(d)
#print(d4)
##print(clf4.coefs_)
#print(clf4.hidden)

#print(accuracy_score(y_train, clf4.predict(X_train)))
#print(accuracy_score(y_test, clf4.predict(X_test)))
#print()


clf5 = copy.deepcopy(clf)
e, d5 = FBI_pruning(clf5, ll, X_train, y_train)
print(e)
print(d5)
#print(clf5.coefs_)
print(clf5.hidden)

print(accuracy_score(y_train, clf5.predict(X_train)))
print(accuracy_score(y_test, clf5.predict(X_test)))
print()


clf6 = copy.deepcopy(clf)
f, d6 = APERT_pruning(clf6, ll, X_train, y_train)
print(f)
print(d6)
#print(clf6.coefs_)
print(clf6.hidden)

print(accuracy_score(y_train, clf6.predict(X_train)))
print(accuracy_score(y_test, clf6.predict(X_test)))
print()


clf7 = copy.deepcopy(clf)
g, d7 = APERTP_pruning(clf7, ll, X_train, y_train)
print(g)
print(d7)
#print(clf7.coefs_)
print(clf7.hidden)

print(accuracy_score(y_train, clf7.predict(X_train)))
print(accuracy_score(y_test, clf7.predict(X_test)))
print()


clf8 = copy.deepcopy(clf)
h, d8 = PD_pruning(clf8, ll, X_train, y_train)
print(h)
print(d8)
#print(clf8.coefs_)
print(clf8.hidden)

print(accuracy_score(y_train, clf8.predict(X_train)))
print(accuracy_score(y_test, clf8.predict(X_test)))
print()


clf9 = copy.deepcopy(clf)
i, d9 = PD_pruning(clf9, ll, X_train, y_train)
print(i)
print(d9)
#print(clf9.coefs_)
print(clf9.hidden)

print(accuracy_score(y_train, clf9.predict(X_train)))
print(accuracy_score(y_test, clf9.predict(X_test)))
print()



x = np.sort(np.random.uniform(-2,2,20)).reshape(-1,1)
y = 2*x + 1

reg = MLP_reg(activation="sigmoid")
reg.fit(x,y)
#print(reg.coefs_)
print(mean_squared_error(y, reg.predict(x)))

reg1 = copy.deepcopy(reg)
aa, dd1 = PEB_pruning(reg1, 0.15, x, y)
print(aa)
print(dd1)
print(reg1.hidden)

print(mean_squared_error(y, reg1.predict(x)))