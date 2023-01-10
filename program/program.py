import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class MLP_class:
    def __init__(self, hidden=(10,10,10), epochs=250, eta=0.1, shuffle=True, part=None, calc_s=False):
        self.hidden = [hidden] if type(hidden)==int else hidden    #Liczba neuronów na kolejnych warstwach ukrytych
        self.layers_count = 2 if type(hidden)==int else len(hidden)+1 #Liczba pow³ok ukrytych
        self.epochs = epochs    #Liczba epok
        self.eta = eta          #Wspó³czynnik uczenia
        self.shuffle = shuffle  #Czy mieszaæ próbki w epokach
        self.part = part        #na jakiej czesci probek uczyæ
        self.coefs_ = None      #wagi
        self.intercepts_ = None #biasy
        self.class_labels_ = None #nazwy klas
        self.class_count = None   #liczba klas
        self.calc_s = calc_s      #czy obliczaæ parametr potrzebny w przycinaniu Karnina
        
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def _forward(self, X):
        activation = [None]*self.layers_count
        tmp = X.copy()
        for i in range(self.layers_count):
            sum_out = np.dot(tmp, self.coefs_[i]) + self.intercepts_[i]
            tmp = self._sigmoid(sum_out)
            activation[i] = tmp
        return activation
    
    def fit(self, X, Y):
        samples, features = X.shape
        if self.class_labels_ is None:
            self.class_labels_ = np.unique(Y)
            self.class_count = len(self.class_labels_)
        y = np.zeros((samples, self.class_count))
        for i in range(samples):
            y[i,np.where(self.class_labels_==Y[i])[0]] = 1
            
        self.coefs_ = [None]*self.layers_count
        self.intercepts_ = [None]*self.layers_count
        self.coefs_[0] = np.random.normal(size=(features,self.hidden[0]))
        self.intercepts_[0] = np.random.normal(size=self.hidden[0])
        for i in range(1,self.layers_count-1):
            self.coefs_[i] = np.random.normal(size=(self.hidden[i-1],self.hidden[i]))
            self.intercepts_[i] = np.random.normal(size=self.hidden[i])
        self.coefs_[-1] = np.random.normal(size=(self.hidden[-1],self.class_count))
        self.intercepts_[-1] = np.random.normal(size=self.class_count)
        
        for epoch in range(self.epochs):
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
            if np.sum(self.predict(X)==Y) == samples:
                print(f"Uczenie zakoñczone po {epoch+1} epokach. Osi¹gniêto dopasowanie.")
                break
    
    def predict(self, X):
        Y = np.zeros(X.shape[0])
        for i, w in enumerate(self._forward(X)[-1]):
            Y[i] = self.class_labels_[np.argmax(w)]
        return Y
    
    def predict_proba(self, X):
        return self._forward(X)[-1]
    
    def get_number_of_parametrs(self):
        res = 0
        for i in range(self.layers_count):
            res += (self.intercepts_[i].shape[0] + self.coefs_[i].shape[0]*self.coefs_[i].shape[1])
        return res