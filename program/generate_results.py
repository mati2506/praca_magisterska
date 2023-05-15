import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import myMLP
import pickle

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

#[X_train, X_test, X_val, y_train, y_test, y_val] = unpickle_all(DATA_FOLDER+f"{data[data_number]}_data.bin")

methods = ['SP', 'SPA', 'KP', 'PBV', 'FBI', 'APERT', 'APERTP', 'PD', 'PEB']

l_n = str(networks_neurons[network_number]) if type(networks_neurons[network_number]) == int else '-'.join(np.array(networks_neurons[network_number], dtype=str))

arch_po_przy = []
for met in methods:
    #if met == 'APERTP': #metoda działa identycznie do APERT dla regresji, więc zostaje pominięta
    #    continue
    #for los in [0, 0.5, 1, 1.75, 2.5]:
    for los in [0, 0.025, 0.05, 0.075, 0.1]:
        [siec] =  unpickle_all(PRUNE_NET_FOLDER+f"{data[data_number]}_network_{l_n}_pruned_{met}_los_{los}")
        arch_po_przy.append(str(siec.hidden) if type(siec.hidden) == int else '-'.join(np.array(siec.hidden, dtype=str)))

arr_in = np.loadtxt(RESULT_FOLDER+f"raw_txt/{data[data_number]}_network_{l_n}.txt", dtype='<U16', delimiter="; ")
df = pd.DataFrame({"Maksymalna utrata":arr_in[:,1], "Metoda":arr_in[:,0],
                   "Czas przycinania":[np.round(np.sum(np.array(row.split(":")).astype(float)*sek_na),3) for row in arr_in[:,2]],
                   "Usunięte połączenia":arr_in[:,4], "Usunięte neurony":arr_in[:,5],
                   "Dokładność":arr_in[:,6], "F1":arr_in[:,7], "Architektura":arch_po_przy})
df['Maksymalna utrata'] = pd.to_numeric([el[:-1] for el in df['Maksymalna utrata']])
df['Usunięte neurony'] = pd.to_numeric(df['Usunięte neurony'])
df['Dokładność'] = pd.to_numeric(df['Dokładność'])
df['F1'] = pd.to_numeric(df['F1'])
df['Usunięte połączenia'] = [np.nan if el == '-' else int(el) for el in df['Usunięte połączenia']]

#tu zrobić ploty

sorterIndex = dict(zip(methods, range(len(methods))))
df['met_sort'] = df["Metoda"].map(sorterIndex)
df.sort_values(by=["Maksymalna utrata", "met_sort"], inplace=True)
df.drop(['met_sort'], axis=1, inplace=True)
df["Maksymalna utrata"] = [str(el)+"%" for el in df['Maksymalna utrata']]
df['Usunięte połączenia'] = ["-" if np.isnan(el) else str(int(el)) for el in df['Usunięte połączenia']]

df.to_csv(RESULT_FOLDER+f"csv/{data[data_number]}_network_{l_n}.csv")
df.to_latex(RESULT_FOLDER+f"tables/{data[data_number]}_network_{l_n}.txt", na_rep="-",
            column_format="|c|c|c|c|c|c|c|c|", longtable=True, index=False)