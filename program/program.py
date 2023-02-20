import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score

class MLP_clf:
    def __init__(self, hidden=(10,10,10), epochs=250, eta=0.1, shuffle=True, part=None, calc_s=True):
        self.hidden = [hidden] if type(hidden)==int else list(hidden)    #Liczba neuronów na kolejnych warstwach ukrytych
        self.layers_count = 2 if type(hidden)==int else len(hidden)+1 #Liczba warstw (ukrytych + wyjściowa)
        self.epochs = epochs    #Liczba epok
        self.eta = eta          #Współczynnik uczenia
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
        return x/np.sum(x)
    
    def _forward(self, X):
        activation = [None]*self.layers_count
        tmp = X.copy()
        for i in range(self.layers_count):
            sum_out = np.dot(tmp, self.coefs_[i]) + self.intercepts_[i]
            tmp = self._sigmoid(sum_out)
            activation[i] = tmp
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
                delta = (activation[-1] - y[i])*(activation[-1]*(1 - activation[-1]))
                gradient = np.outer(activation[-2], delta)
                self.intercepts_[-1] -= self.eta*delta
                for j in range(self.layers_count-2,0,-1):
                    delta = np.dot(delta, self.coefs_[j+1].T)*(activation[j]*(1 - activation[j]))
                    self.coefs_[j+1] -= self.eta*gradient
                    gradient = np.outer(activation[j-1], delta)
                    self.intercepts_[j] -= self.eta*delta
                delta = np.dot(delta, self.coefs_[1].T)*(activation[0]*(1 - activation[0]))
                self.coefs_[1] -= self.eta*gradient
                gradient = np.outer(X[i], delta)
                self.intercepts_[0] -= self.eta*delta
                self.coefs_[0] -= self.eta*gradient

            if self.calc_s:
                for i in range(self.layers_count):
                    coef = self.coefs_[i]
                    if np.all(coef != init_coefs[i]):
                        self.karnin_s[i] += ((coef-last_coefs[i])**2)*(coef/(self.eta*(coef-init_coefs[i])))

            if np.sum(self.predict(X_val)==Y_val) == samples:
                print(f"Uczenie zakończone po {epoch+1} epokach. Osiągnięto dopasowanie.")
                break
    
    def predict(self, X):
        Y = np.zeros(X.shape[0], dtype=self.class_type)
        for i, w in enumerate(X):
            Y[i] = self.class_labels_[np.argmax(self._forward(w)[-1])]
        return Y
    
    def predict_proba(self, X):
        return np.array([self._forward(x)[-1] for x in X])
    
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



class MLP_reg:
    def __init__(self, hidden=(10,10,10), epochs=250, eta=0.1, shuffle=True, part=None, calc_s=True):
        self.hidden = [hidden] if type(hidden)==int else list(hidden)    #Liczba neuronów na kolejnych warstwach ukrytych
        self.layers_count = 2 if type(hidden)==int else len(hidden)+1 #Liczba warstw (ukrytych + wyjściowa)
        self.epochs = epochs    #Liczba epok
        self.eta = eta          #Współczynnik uczenia
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
            tmp = self._sigmoid(sum_out)
            activation[i] = tmp
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
                delta = (activation[-1] - Y[i])*(activation[-1]*(1 - activation[-1]))
                gradient = np.outer(activation[-2], delta)
                self.intercepts_[-1] -= self.eta*delta
                for j in range(self.layers_count-2,0,-1):
                    delta = np.dot(delta, self.coefs_[j+1].T)*(activation[j]*(1 - activation[j]))
                    self.coefs_[j+1] -= self.eta*gradient
                    gradient = np.outer(activation[j-1], delta)
                    self.intercepts_[j] -= self.eta*delta
                delta = np.dot(delta, self.coefs_[1].T)*(activation[0]*(1 - activation[0]))
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



def simple_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True): #lost - maksymalna procentowa utrata dokładności podczas przycinania
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

    tmp_w = copy.deepcopy(clf_reg.coefs_)
    tmp_b = copy.deepcopy(clf_reg.intercepts_)

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
                sums = np.nansum(tmp_w[i], axis=1)
                wh = np.where(sums==0)[0]
                for ind in wh: #for wykonywany jednokrotnie
                    tmp_w[i] = np.delete(tmp_w[i], ind, 0)
                    tmp_w[i-1] = np.delete(tmp_w[i-1], ind, 1)
                    tmp_b[i-1] = np.delete(tmp_b[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi, może gdzieś tą wartośc by dodawać???
                    clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                    clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                    clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi, może gdzieś tą wartośc by dodawać???

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
        #clf_reg.refit(X_t, y_t, X_v, y_v) #NAPISAĆ METODĘ DOSTARAJAJĄCĄ PO PRZYCINANIU!!!
    return del_w

def _outs_of_single_neuron(clf_reg, X, weight, bias, number, index):
    w = X.shape[0]
    outs = np.zeros(w)
    for j in range(w):
        activation_i = X[j].copy()
        for i in range(number):
            sum_out = np.dot(activation_i, weight[i]) + bias[i]
            activation_i = clf_reg._sigmoid(sum_out)
            outs[i] = activation_i[index[0]]*weight[number][index]
    return outs

   

#test działania
data = pd.read_csv("iris.data")
X = data.iloc[:,1:4].values
Y = data.iloc[:,4].values
X_train, X_test, y_train, y_test = train_test_split(X, Y)
clf = MLP_clf(epochs=100)
clf.fit(X_train, y_train)
print(clf.coefs_)

a = simple_pruning(clf, 0.5, X_train, y_train)
print(a)
print(clf.coefs_)

print(clf.predict(X_test))
print(y_test)