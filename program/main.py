import numpy as np
import pandas as pd
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import normalize
import myMLP
import myNetworkPruning as prune
import pickle


RAW_DATA_FOLDER = "./raw_data/"
DATA_FOLDER = "./data/"
NETWORK_FOLDER = "./networks/"
PRUNE_NET_FOLDER = "./pruned_networks/"
RESULT_FOLDER = "./results/"


def pickle_all(fname, some_list):
    f = open(fname, "wb+")
    pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def unpickle_all(fname): 
    f = open(fname, "rb")
    some_list = pickle.load(f)
    f.close()
    return some_list


#PRZYGOTOWANIE DANYCH - normalizacja; podział zbioru na train, test i validation w stosunku (0.6, 0.3, 0.1)
#rice - klasyfikacja
#data = pd.read_csv(RAW_DATA_FOLDER+"riceClassification.csv")
#X = normalize(data.iloc[:,1:-1].values, norm="max", axis=0)
#Y = data.iloc[:,-1].values
#X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, train_size=0.6)
#X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)
#pickle_all(DATA_FOLDER+"rice_data.bin", [X_train, X_test, X_val, y_train, y_test, y_val])

#anuran - klasyfikacja
#data = pd.read_csv(RAW_DATA_FOLDER+"Frogs_MFCCs.csv")
#X = normalize(data.iloc[:,:-4].values, norm="max", axis=0)
#Y = data.iloc[:,-4:-1].values
#X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, train_size=0.6)
#X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)
#pickle_all(DATA_FOLDER+"anuran_family_data.bin", [X_train, X_test, X_val, y_train[:,0], y_test[:,0], y_val[:,0]])
#pickle_all(DATA_FOLDER+"anuran_genus_data.bin", [X_train, X_test, X_val, y_train[:,1], y_test[:,1], y_val[:,1]])
#pickle_all(DATA_FOLDER+"anuran_species_data.bin", [X_train, X_test, X_val, y_train[:,2], y_test[:,2], y_val[:,2]])

#Dry_Bean - klasyfikacja
#data = pd.read_excel(RAW_DATA_FOLDER+"Dry_Bean_Dataset.xlsx")
#X = normalize(data.iloc[:,:-1].values, norm="max", axis=0)
#Y = data.iloc[:,-1].values
#X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, train_size=0.6)
#X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)
#pickle_all(DATA_FOLDER+"dry_bean_data.bin", [X_train, X_test, X_val, y_train, y_test, y_val])

#Electrical_Grid - regresja
#data = pd.read_csv(RAW_DATA_FOLDER+"Electrical_Grid.csv")
#X = normalize(data.iloc[:,:-2].values, norm="max", axis=0)
#Y = data.iloc[:,-2].values
#X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, train_size=0.6)
#X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)
#pickle_all(DATA_FOLDER+"electrical_grid_data.bin", [X_train, X_test, X_val, y_train, y_test, y_val])

#Parkinsons_Telemonitoring
#data = pd.read_csv(RAW_DATA_FOLDER+"parkinsons_updrs.data")
#X = normalize(data.iloc[:,6:].values, norm="max", axis=0)
#Y = data.iloc[:,4:6].values
#X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, train_size=0.6)
#X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)
#pickle_all(DATA_FOLDER+"parkinson_motor_data.bin", [X_train, X_test, X_val, y_train[:,0], y_test[:,0], y_val[:,0]])
#pickle_all(DATA_FOLDER+"parkinson_total_data.bin", [X_train, X_test, X_val, y_train[:,1], y_test[:,1], y_val[:,1]])

#Gas_Turbine
#data = np.loadtxt(RAW_DATA_FOLDER+"propulsion_plants.txt")
#X = normalize(data[:,:-2], norm="max", axis=0)
#Y = data[:,-2:]
#X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, train_size=0.6)
#X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)
#pickle_all(DATA_FOLDER+"GT_compressor_data.bin", [X_train, X_test, X_val, y_train[:,0], y_test[:,0], y_val[:,0]])
#pickle_all(DATA_FOLDER+"GT_turbine_data.bin", [X_train, X_test, X_val, y_train[:,1], y_test[:,1], y_val[:,1]])


data = ["rice", "anuran_family", "anuran_genus", "anuran_species", "dry_bean",
        "electrical_grid", "parkinson_motor", "parkinson_total", "GT_compressor", "GT_turbine"] #nazwy plików zbiorów danych
networks_neurons = [(30), (18,15), (16,13,10)]

data_number = 9 #numer danych, na których będzie aktualne uruchomienie programu
network_number = 0 #numer architektury sieci, dla której będzie aktualne uruchomienie programu


#ODCZYT DANYCH Z PLIKU .BIN
[X_train, X_test, X_val, y_train, y_test, y_val] = unpickle_all(DATA_FOLDER+f"{data[data_number]}_data.bin")


#PRZYGOTOWANIE SIECI
#Dla klasyfikacji
#for hidden in networks_neurons:
#    clf = myMLP.Classifier(hidden=hidden, epochs=500)
#    t1 = time.time()
#    clf.fit(X_train, y_train, X_val, y_val)
#    t_t = str(timedelta(seconds=(time.time()-t1)))
#    f1_train = f1_score(y_train, clf.predict(X_train), average='macro')
#    f1_test = f1_score(y_test, clf.predict(X_test), average='macro')
#    f1_val = f1_score(y_val, clf.predict(X_val), average='macro')
#    l_n = str(hidden) if type(hidden) == int else '-'.join(np.array(hidden, dtype=str))
#    f = open("nauczone_sieci.txt", 'a')
#    f.write(f"{data[data_number]}_network_{l_n}: {t_t}  f1_train: {f1_train}  f1_test: {f1_test}  f1_validation: {f1_val} \n")
#    f.close()
#    pickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin", [clf])    
#    print(t_t)

#Dla regresji
for hidden in networks_neurons:
    reg = myMLP.Regressor(hidden=hidden, epochs=500)
    t1 = time.time()
    reg.fit(X_train, y_train, X_val, y_val)
    t_t = str(timedelta(seconds=(time.time()-t1)))
    MSE_train = mean_squared_error(y_train, reg.predict(X_train))
    MSE_test = mean_squared_error(y_test, reg.predict(X_test))
    MSE_val = mean_squared_error(y_val, reg.predict(X_val))
    l_n = str(hidden) if type(hidden) == int else '-'.join(np.array(hidden, dtype=str))
    f = open("nauczone_sieci.txt", 'a')
    f.write(f"{data[data_number]}_network_{l_n}: {t_t}  MSE_train: {MSE_train}  MSE_test: {MSE_test}  MSE_validation: {MSE_val} \n")
    f.close()
    pickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin", [reg])    
    print(t_t)


#ODCZYT SIECI Z PLIKU .BIN
#l_n = str(networks_neurons[network_number]) if type(networks_neurons[network_number]) == int else '-'.join(np.array(networks_neurons[network_number], dtype=str))
#[clf] = unpickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin")
#[reg] = unpickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin")

#CO ZBIERAĆ Z TESTÓW:
#poziom zakładanej utraty f1 (np. 0%, 2.5%, 5%, 7.5%, 10%) lub wzrostu MSE (np. 0%, 50%, 100%, 200%, 500%)
#czas przycinania
#wartości f1/dokładności/macierz_konfuzji/? dla klasyfikacji; MSE/MAE/? dla regresji
#liczba usuniętych wag/neuronów
#czas predykcji po przycinaniu???
#...???
   

#test działania
##data = pd.read_csv(RAW_DATA_FOLDER+"iris.data", header=None)
##X = normalize(data.iloc[:,0:4].values, norm="max", axis=0)
##Y = data.iloc[:,4].values
##X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, train_size=0.6)
##X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)

##pickle_all(DATA_FOLDER+"iris_data.bin", [X_train, X_test, X_val, y_train, y_test, y_val])
#[X_train, X_test, X_val, y_train, y_test, y_val] = unpickle_all(DATA_FOLDER+"iris_data.bin")

##clf = myMLP.Classifier(epochs=100, activation="sigmoid")
##clf.fit(X_train, y_train, X_val, y_val)

##pickle_all(NETWORK_FOLDER+"iris.bin", [clf])
#[clf] = unpickle_all(NETWORK_FOLDER+"iris.bin")

##print(clf.coefs_)
#print("Sieć bez przycięcia:")
#print("F1 train: ", f1_score(y_train, clf.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf.predict(X_val), average='macro'))
#print()

#ll = 0.05

#print("simple_pruning:")
#clf1 = copy.deepcopy(clf)
#a, d1, t1 = prune.simple_pruning(clf1, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("[Przycięte wagi, neurony]: ", a)
#print("F1 przed douczaniem", d1)
#print("Czas przycinania: ", t1)
##print(clf1.coefs_)
#print("Architektura po przycinaniu:", clf1.hidden)

#print("F1 train: ", f1_score(y_train, clf1.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf1.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf1.predict(X_val), average='macro'))
#print()


#print("simple_pruning_amendment:")
#clf2 = copy.deepcopy(clf)
#b, d2, t2 = prune.simple_pruning_amendment(clf2, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("[Przycięte wagi, neurony]: ", b)
#print("F1 przed douczaniem", d2)
#print("Czas przycinania: ", t2)
##print(clf2.coefs_)
#print("Architektura po przycinaniu:", clf2.hidden)

#print("F1 train: ", f1_score(y_train, clf2.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf2.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf2.predict(X_val), average='macro'))
#print()


#print("karnin_pruning:")
#clf3 = copy.deepcopy(clf)
#c, d3, t3 = prune.karnin_pruning(clf3, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("[Przycięte wagi, neurony]: ", c)
#print("F1 przed douczaniem", d3)
#print("Czas przycinania: ", t3)
##print(clf3.coefs_)
#print("Architektura po przycinaniu:", clf3.hidden)

#print("F1 train: ", f1_score(y_train, clf3.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf3.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf3.predict(X_val), average='macro'))
#print()


#print("pruning_by_variance:")
#clf4 = copy.deepcopy(clf)
#d, d4, t4 = prune.pruning_by_variance(clf4, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("[Przycięte wagi, neurony]: ", d)
#print("F1 przed douczaniem", d4)
#print("Czas przycinania: ", t4)
##print(clf4.coefs_)
#print("Architektura po przycinaniu:", clf4.hidden)

#print("F1 train: ", f1_score(y_train, clf4.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf4.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf4.predict(X_val), average='macro'))
#print()


#print("FBI_pruning:")
#clf5 = copy.deepcopy(clf)
#e, d5, t5 = prune.FBI_pruning(clf5, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("Przycięte neurony:", e)
#print("F1 przed douczaniem", d5)
#print("Czas przycinania: ", t5)
##print(clf5.coefs_)
#print("Architektura po przycinaniu:", clf5.hidden)

#print("F1 train: ", f1_score(y_train, clf5.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf5.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf5.predict(X_val), average='macro'))
#print()


#print("APERT_pruning:")
#clf6 = copy.deepcopy(clf)
#f, d6, t6 = prune.APERT_pruning(clf6, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("Przycięte neurony:", f)
#print("F1 przed douczaniem", d6)
#print("Czas przycinania: ", t6)
##print(clf6.coefs_)
#print("Architektura po przycinaniu:", clf6.hidden)

#print("F1 train: ", f1_score(y_train, clf6.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf6.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf6.predict(X_val), average='macro'))
#print()


#print("APERTP_pruning:")
#clf7 = copy.deepcopy(clf)
#g, d7, t7 = prune.APERTP_pruning(clf7, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("Przycięte neurony:", g)
#print("F1 przed douczaniem", d7)
#print("Czas przycinania: ", t7)
##print(clf7.coefs_)
#print("Architektura po przycinaniu:", clf7.hidden)

#print("F1 train: ", f1_score(y_train, clf7.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf7.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf7.predict(X_val), average='macro'))
#print()


#print("PD_pruning:")
#clf8 = copy.deepcopy(clf)
#h, d8, t8 = prune.PD_pruning(clf8, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("Przycięte neurony:", h)
#print("F1 przed douczaniem", d8)
#print("Czas przycinania: ", t8)
##print(clf8.coefs_)
#print("Architektura po przycinaniu:", clf8.hidden)

#print("F1 train: ", f1_score(y_train, clf8.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf8.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf8.predict(X_val), average='macro'))
#print()


#print("PEB_pruning:")
#clf9 = copy.deepcopy(clf)
#i, d9, t9 = prune.PEB_pruning(clf9, ll, X_train, y_train, X_v=X_val, y_v=y_val)
#print("Przycięte neurony:", i)
#print("F1 przed douczaniem", d9)
#print("Czas przycinania: ", t9)
##print(clf9.coefs_)
#print("Architektura po przycinaniu:", clf9.hidden)

#print("F1 train: ", f1_score(y_train, clf9.predict(X_train), average='macro'))
#print("F1 test: ", f1_score(y_test, clf9.predict(X_test), average='macro'))
#print("F1 validation ", f1_score(y_val, clf9.predict(X_val), average='macro'))
#print()



#x = np.sort(np.random.uniform(-2,2,250)).reshape(-1,1)
#y = 2*x + 1

#reg = myMLP.Regressor(activation="sigmoid")
#reg.fit(x,y)
##print(reg.coefs_)
#print(mean_squared_error(y, reg.predict(x)))

#reg1 = copy.deepcopy(reg)
#aa, dd1, tt1 = prune.FBI_pruning(reg1, 1, x, y)
#print(aa)
#print(dd1)
#print(tt1)
#print(reg1.hidden)

#print(mean_squared_error(y, reg1.predict(x)))