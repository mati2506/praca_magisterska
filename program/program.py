import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error

class MLP_clf:
    def __init__(self, hidden=(10,10,10), epochs=250, eta=0.1, shuffle=True, part=None, calc_s=True):
        self.hidden = [hidden] if type(hidden)==int else hidden    #Liczba neuronów na kolejnych warstwach ukrytych
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

        if X_val is None and Y_val is None:
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
        self.hidden = [hidden] if type(hidden)==int else hidden    #Liczba neuronów na kolejnych warstwach ukrytych
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
        if X_val is None and Y_val is None:
            X_val = X.copy()
            Y_val = Y.copy()
            
        self.coefs_ = [None]*self.layers_count
        self.intercepts_ = [None]*self.layers_count
        self.coefs_[0] = np.random.normal(size=(features,self.hidden[0]))
        self.intercepts_[0] = np.random.normal(size=self.hidden[0])
        for i in range(1,self.layers_count-1):
            self.coefs_[i] = np.random.normal(size=(self.hidden[i-1],self.hidden[i]))
            self.intercepts_[i] = np.random.normal(size=self.hidden[i])
        self.coefs_[-1] = np.random.normal(size=(self.hidden[-1]))
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




#test działania
data = pd.read_csv("iris.data")
X = data.iloc[:,1:4].values
Y = data.iloc[:,4].values
X_train, X_test, y_train, y_test = train_test_split(X, Y)
clf = MLP_clf(epochs=100)
clf.fit(X_train, y_train)
print(clf.coefs_)

print(clf.predict(X_test))
print(y_test)