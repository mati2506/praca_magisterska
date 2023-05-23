import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import myMLP
import pickle
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.close('all')

DATA_FOLDER = "./data/"
NETWORK_FOLDER = "./networks/"
PRUNE_NET_FOLDER = "./pruned_networks/"
RESULT_FOLDER = "./results/"

def unpickle_all(fname): 
    f = open(fname, "rb")
    some_list = pickle.load(f)
    f.close()
    return some_list

sek_na = np.array([3600, 60, 1])

data = ["rice", "anuran_family", "anuran_genus", "anuran_species", "dry_bean",
        "electrical_grid", "parkinson_motor", "parkinson_total", "GT_compressor", "GT_turbine"] #nazwy plików zbiorów danych
networks_neurons = [(30), (18,15), (16,13,10)]

data_number = 0
network_number = 0


# dok_f1_table = []
# for data_number in [0,3,4]:
#         [X_train, X_test, X_val, y_train, y_test, y_val] = unpickle_all(DATA_FOLDER+f"{data[data_number]}_data.bin")
#         for network_number in range(3):
#             l_n = str(networks_neurons[network_number]) if type(networks_neurons[network_number]) == int else '-'.join(np.array(networks_neurons[network_number], dtype=str))
#             [clf] = unpickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin")
#             y_pred_train = clf.predict(X_train)
#             y_pred_test = clf.predict(X_test)
#             acc_train = np.round(accuracy_score(y_train, y_pred_train), 3)
#             f1_train = np.round(f1_score(y_train, y_pred_train, average='macro'), 3)
#             acc_test = np.round(accuracy_score(y_test, y_pred_test), 3)
#             f1_test = np.round(f1_score(y_test, y_pred_test, average='macro'), 3)
#             dok_f1_table.append([data[data_number], l_n, acc_train, f1_train, acc_test, f1_test])
# df = pd.DataFrame(dok_f1_table, columns=["Zbiór", "Architektura", "dok_ucz", "f1_ucz", "dok_test", "f1_test"])
# df.set_index(["Zbiór", "Architektura"], inplace=True)
# df.to_csv(RESULT_FOLDER+"klasyfikacje.csv")
# df.to_latex(RESULT_FOLDER+"klasyfikacje.txt", column_format="|c|c|c|c|c|c|")

# mse_r2_table = []
# for data_number in [5,8,9]:
#         [X_train, X_test, X_val, y_train, y_test, y_val] = unpickle_all(DATA_FOLDER+f"{data[data_number]}_data.bin")
#         for network_number in range(3):
#             l_n = str(networks_neurons[network_number]) if type(networks_neurons[network_number]) == int else '-'.join(np.array(networks_neurons[network_number], dtype=str))
#             [clf] = unpickle_all(NETWORK_FOLDER+f"{data[data_number]}_network_{l_n}.bin")
#             y_pred_train = clf.predict(X_train)
#             y_pred_test = clf.predict(X_test)
#             acc_train = np.round(mean_squared_error(y_train, y_pred_train), 6)
#             f1_train = np.round(r2_score(y_train, y_pred_train), 3)
#             acc_test = np.round(mean_squared_error(y_test, y_pred_test), 6)
#             f1_test = np.round(r2_score(y_test, y_pred_test), 3)
#             mse_r2_table.append([data[data_number], l_n, acc_train, f1_train, acc_test, f1_test])
# df = pd.DataFrame(mse_r2_table, columns=["Zbiór", "Architektura", "mse_ucz", "R^{2}_ucz", "mse_test", "R^{2}_test"])
# df.set_index(["Zbiór", "Architektura"], inplace=True)
# df.to_csv(RESULT_FOLDER+"regresje.csv")
# df.to_latex(RESULT_FOLDER+"regresje.txt", column_format="|c|c|c|c|c|c|")




methods = ['SP', 'SPA', 'KP', 'PBV', 'FBI', 'APERT', 'APERTP', 'PD', 'PEB']

l_n = str(networks_neurons[network_number]) if type(networks_neurons[network_number]) == int else '-'.join(np.array(networks_neurons[network_number], dtype=str))

arch_po_przy = []
for met in methods:
    # if met == 'APERTP': #metoda działa identycznie do APERT dla regresji, więc zostaje pominięta
    #     continue
    # for los in [0, 0.5, 1, 1.75, 2.5]:
    for los in [0, 0.025, 0.05, 0.075, 0.1]:
        [siec] =  unpickle_all(PRUNE_NET_FOLDER+f"{data[data_number]}_network_{l_n}_pruned_{met}_los_{los}")
        arch_po_przy.append(str(siec.hidden) if type(siec.hidden) == int else '-'.join(np.array(siec.hidden, dtype=str)))

arr_in = np.loadtxt(RESULT_FOLDER+f"raw_txt/{data[data_number]}_network_{l_n}.txt", dtype='<U16', delimiter="; ")
df = pd.DataFrame({"MUJ":arr_in[:,1], "Metoda":arr_in[:,0],
                    "CP":[np.round(np.sum(np.array(row.split(":")).astype(float)*sek_na),3) for row in arr_in[:,2]],
                    "UP":arr_in[:,4], "UN":arr_in[:,5],
                    "Dok":arr_in[:,6], "F1":arr_in[:,7], "APP":arch_po_przy})
# df = pd.DataFrame({"MUJ":arr_in[:,1], "Metoda":arr_in[:,0],
#                     "CP":[np.round(np.sum(np.array(row.split(":")).astype(float)*sek_na),3) for row in arr_in[:,2]],
#                     "UP":arr_in[:,4], "UN":arr_in[:,5],
#                     "MSE":arr_in[:,6], "R^{2}":arr_in[:,7], "APP":arch_po_przy})
df['MUJ'] = pd.to_numeric([el[:-1] for el in df['MUJ']])
df['UN'] = pd.to_numeric(df['UN'])
df['Dok'] = pd.to_numeric(df['Dok'])
df['F1'] = pd.to_numeric(df['F1'])
# df['MSE'] = pd.to_numeric(df['MSE'])
# df['R^{2}'] = pd.to_numeric(df['R^{2}'])
df['UP'] = [np.nan if el == '-' else int(el) for el in df['UP']]

#ploty
plt.figure(figsize=(16,9))
for i in np.unique(df['Metoda']):
    wh = np.where(df['Metoda'].values == i)[0]
    plt.plot(df['MUJ'].values[wh], df['UN'].values[wh], 'o-', label=i)
plt.legend()
plt.xlabel("Maksymalna utrata jakości")
plt.ylabel("Usunięte neurony")
plt.title("Usunięte neurony w zależności od utraty jakości")
plt.savefig(RESULT_FOLDER+f"plots/UN_{data[data_number]}_network_{l_n}.png")

plt.figure(figsize=(16,9))
for i in np.unique(df['Metoda']):
    wh = np.where(df['Metoda'].values == i)[0]
    plt.plot(df['MUJ'].values[wh], df['CP'].values[wh], 'o-', label=i)
plt.legend()
plt.xlabel("Maksymalna utrata jakości")
plt.ylabel("Czas trwania przycinania [s]")
plt.title("Czas przycinania w zależności od utraty jakości")
plt.savefig(RESULT_FOLDER+f"plots/CP_{data[data_number]}_network_{l_n}.png")

'''
#Generowanie tabeli latex
sorterIndex = dict(zip(methods, range(len(methods))))
df['met_sort'] = df["Metoda"].map(sorterIndex)
df.sort_values(by=["MUJ", "met_sort"], inplace=True)
df.drop(['met_sort'], axis=1, inplace=True)
df["MUJ"] = [str(el)+"%" for el in df['MUJ']]
df['UP'] = ["-" if np.isnan(el) else str(int(el)) for el in df['UP']]
df.set_index(["MUJ", "Metoda"], inplace=True)

df.to_csv(RESULT_FOLDER+f"csv/{data[data_number]}_network_{l_n}.csv")
df.to_latex(RESULT_FOLDER+f"tables/{data[data_number]}_network_{l_n}.txt",
              column_format="|c|c|c|c|c|c|c|c|", longtable=True)
'''