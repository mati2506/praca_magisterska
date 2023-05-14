import numpy as np
import pandas as pd
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import normalize
import myMLP
import myNetworkPruning as prune
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

data_number = 6 #numer danych, na których będzie aktualne uruchomienie programu
network_number = 0 #numer architektury sieci, dla której będzie aktualne uruchomienie programu (tylko dla uczenia)


#ODCZYT DANYCH Z PLIKU .BIN
[X_train, X_test, X_val, y_train, y_test, y_val] = unpickle_all(DATA_FOLDER+f"{data[data_number]}_data.bin")


#PRZYGOTOWANIE SIECI
#Dla klasyfikacji
#for hidden in networks_neurons:
#    clf = myMLP.Classifier(hidden=hidden, epochs=500) #, activation="relu")
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
#for hidden in networks_neurons:
#    reg = myMLP.Regressor(hidden=hidden, epochs=500) #, activation="relu")
#    t1 = time.time()
#    reg.fit(X_train, y_train, X_val, y_val)
#    t_t = str(timedelta(seconds=(time.time()-t1)))
#    MSE_train = mean_squared_error(y_train, reg.predict(X_train))
#    MSE_test = mean_squared_error(y_test, reg.predict(X_test))
#    MSE_val = mean_squared_error(y_val, reg.predict(X_val))
#    l_n = str(hidden) if type(hidden) == int else '-'.join(np.array(hidden, dtype=str))
#    f = open("nauczone_sieci.txt", 'a')
#    f.write(f"{data[data_number]}_network_{l_n}: {t_t}  MSE_train: {MSE_train}  MSE_test: {MSE_test}  MSE_validation: {MSE_val} \n")
#    f.close()
#    pickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin", [reg])    
#    print(t_t)
  

#BADANIA
methods = {'SP':prune.simple_pruning, 'SPA':prune.simple_pruning_amendment, 'KP':prune.karnin_pruning,
          'PBV':prune.pruning_by_variance, 'FBI':prune.FBI_pruning, 'APERT':prune.APERT_pruning, 'APERTP':prune.APERTP_pruning,
          'PD':prune.PD_pruning, 'PEB':prune.PEB_pruning}

##dla klasyfikacji
#for network_number in range(3): #pętla po architekturach sieci
#    print("Architektura", networks_neurons[network_number])
#    l_n = str(networks_neurons[network_number]) if type(networks_neurons[network_number]) == int else '-'.join(np.array(networks_neurons[network_number], dtype=str))
#    [clf] = unpickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin") #odczyt sieci z pliku .bin

#    for met in methods:
#        for los in [0, 0.025, 0.05, 0.075, 0.1]:
#            clf_t = copy.deepcopy(clf)
#            dele, f1_p, t_p = methods[met](clf_t, los, X_train, y_train, X_v=X_val, y_v=y_val, ep=50)
#            pickle_all(PRUNE_NET_FOLDER+f"{data[data_number]}_network_{l_n}_pruned_{met}_los_{los}", [clf_t])
#            print(met, los, dele, f1_p, t_p)
#            t_mean = 0 #średni czas predykcji
#            for _ in range(25):
#                t1 = time.time()
#                y_pred = clf_t.predict(X_test)
#                t_mean += (time.time()-t1)
#            t_mean = t_mean/25
#            f1_t = f1_score(y_test, y_pred, average='macro')
#            acc = accuracy_score(y_test, y_pred)
#            #kolejność kolumn: metoda przycinania, max utrata f1, czas przycinania, f1 przed douczaniem, usunięte połączenia, usunięte neurony, dokładność po przycinaniu, f1 po przycinaniu, czas predykcji po przycinaniu
#            if met in ['SP', 'SPA', 'KP', 'PBV']:
#                row = f"{met}; {los*100}%; {t_p}; {np.round(f1_p,3)}; {dele[0]}; {dele[1]}; {np.round(acc,3)}; {np.round(f1_t,3)}; {np.round(t_mean,6)}s \n"
#            else:
#                row = f"{met}; {los*100}%; {t_p}; {np.round(f1_p,3)}; -; {dele}; {np.round(acc,3)}; {np.round(f1_t,3)}; {np.round(t_mean,6)}s \n"
#            f = open(RESULT_FOLDER+f"raw_txt/{data[data_number]}_network_{l_n}.txt", 'a')
#            f.write(row)
#            f.close()
#            #generowanie macierzy konfuzji
#            cls_names = clf_t.class_labels_ #nazwy klas w zbiorze
#            conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=cls_names), index=cls_names, columns=cls_names)
#            column_format = "|c|" + "".join(["c"]*conf_matrix.shape[0]) + "|"
#            conf_matrix.to_latex(RESULT_FOLDER+f"conf_matrix/{data[data_number]}_network_{l_n}_pruned_{met}_los_{los}.txt", column_format=column_format)

##dla regresji
for network_number in range(3): #pętla po architekturach sieci
    print("Architektura", networks_neurons[network_number])
    l_n = str(networks_neurons[network_number]) if type(networks_neurons[network_number]) == int else '-'.join(np.array(networks_neurons[network_number], dtype=str))
    [reg] = unpickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin") #odczyt sieci z pliku .bin

    for met in methods:
        if met == 'APERTP': #metoda działa identycznie do APERT dla regresji, więc zostaje pominięta
            continue
        for los in [0, 0.5, 1, 1.75, 2.5]:
            reg_t = copy.deepcopy(reg)
            dele, MSE_p, t_p = methods[met](reg_t, los, X_train, y_train, X_v=X_val, y_v=y_val, ep=50)
            pickle_all(PRUNE_NET_FOLDER+f"{data[data_number]}_network_{l_n}_pruned_{met}_los_{los}", [reg_t])
            print(met, los, dele, MSE_p, t_p)
            t_mean = 0 #średni czas predykcji
            for _ in range(25):
                t1 = time.time()
                y_pred = reg_t.predict(X_test)
                t_mean += (time.time()-t1)
            t_mean = t_mean/25
            MSE_t = mean_squared_error(y_test, y_pred)
            R2 = r2_score(y_test, y_pred)
            #kolejność kolumn: metoda przycinania, max wzrost MSE, czas przycinania, MSE przed douczaniem, usunięte połączenia, usunięte neurony, MSE po przycinaniu, R2 po przycinaniu, czas predykcji po przycinaniu
            if met in ['SP', 'SPA', 'KP', 'PBV']:
                row = f"{met}; {los*100}%; {t_p}; {np.round(MSE_p,6)}; {dele[0]}; {dele[1]}; {np.round(MSE_t,6)}; {np.round(R2,3)}; {np.round(t_mean,6)}s \n"
            else:
                row = f"{met}; {los*100}%; {t_p}; {np.round(MSE_p,6)}; -; {dele}; {np.round(MSE_t,6)}; {np.round(R2,3)}; {np.round(t_mean,6)}s \n"
            f = open(RESULT_FOLDER+f"raw_txt/{data[data_number]}_network_{l_n}.txt", 'a')
            f.write(row)
            f.close()


