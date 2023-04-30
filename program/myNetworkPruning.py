import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import copy
import time
from datetime import timedelta
import myPruningMetrics as met

def simple_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_coefs = clf_reg.get_nuber_of_coefs()
    del_w = 0 #liczba usuniętych wag
    del_n = 0 #liczba usuniętych neuronów

    tmp_w = copy.deepcopy(clf_reg.coefs_)

    for i in range(l_c):       
        del_w += np.sum(tmp_w[i] == 0)
        tmp_w[i][tmp_w[i] == 0] = np.nan

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
        if np.all(np.isnan(tmp_val)):
            break
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        tmp_w[tmp][tmp_ind[tmp]] = np.nan
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if tmp_w[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(tmp_w[i], axis=1)
                    for ind in (np.where(sums==0)[0])[::-1]:
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
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar, str(timedelta(seconds=(time.time()-start_time)))

def simple_pruning_amendment(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_coefs = clf_reg.get_nuber_of_coefs()
    del_w = 0 #liczba usuniętych wag
    del_n = 0 #liczba usuniętych neuronów

    tmp_w = copy.deepcopy(clf_reg.coefs_)
    c_r_clc = copy.deepcopy(clf_reg)

    for i in range(l_c):      
        del_w += np.sum(tmp_w[i] == 0)
        tmp_w[i][tmp_w[i] == 0] = np.nan

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
        if np.all(np.isnan(tmp_val)):
            break
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        tmp_w[tmp][tmp_ind[tmp]] = np.nan
        ind_w = list(tmp_ind[tmp])
        ind_w = (ind_w[0]+np.sum(del_n_n[tmp] <= ind_w[0]), ind_w[1]+np.sum(del_n_n[tmp+1] <= ind_w[1]))
        clf_reg.intercepts_[tmp][tmp_ind[tmp][1]] += met.mean_outs_of_single_weight(c_r_clc, X_t, tmp, ind_w)
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if tmp_w[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(tmp_w[i], axis=1)
                    for ind in (np.where(sums==0)[0])[::-1]:
                        del_n_n[i][del_n_n[i] > ind] -= 1 #przesunięcie, aby uniknąć pominięcia usuniętego wiersza, gdy najpierw usuwany jest wiersz o wyższym indeksie, a później o niższym
                        del_n_n[i] = np.append(del_n_n[i], [ind])
                        tmp_w[i] = np.delete(tmp_w[i], ind, 0)
                        tmp_w[i-1] = np.delete(tmp_w[i-1], ind, 1)
                        del_w += np.sum(clf_reg.coefs_[i-1][:,ind] != 0) #usunięte wagi z kolumny warstwy poprzedzającej (potrzebne do zakończenia głównego while)
                        del_n += 1
                        clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                        clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                        clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi
                        if tmp_w[i].shape[0] < 2:
                            break

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar, str(timedelta(seconds=(time.time()-start_time)))

def karnin_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        if np.all(np.isnan(tmp_val)):
            break
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        s[tmp][tmp_ind[tmp]] = np.nan
        ind_w = list(tmp_ind[tmp])
        ind_w = (ind_w[0]+np.sum(del_n_n[tmp] <= ind_w[0]), ind_w[1]+np.sum(del_n_n[tmp+1] <= ind_w[1]))
        clf_reg.intercepts_[tmp][tmp_ind[tmp][1]] += met.mean_outs_of_single_weight(c_r_clc, X_t, tmp, ind_w)
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if s[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(s[i], axis=1)
                    for ind in (np.where(sums==0)[0])[::-1]:
                        del_n_n[i][del_n_n[i] > ind] -= 1 #przesunięcie, aby uniknąć pominięcia usuniętego wiersza, gdy najpierw usuwany jest wiersz o wyższym indeksie, a później o niższym
                        del_n_n[i] = np.append(del_n_n[i], [ind])
                        s[i] = np.delete(s[i], ind, 0)
                        s[i-1] = np.delete(s[i-1], ind, 1)
                        del_w += np.sum(clf_reg.coefs_[i-1][:,ind] != 0) #usunięte wagi z kolumny warstwy poprzedzającej (potrzebne do zakończenia głównego while)
                        del_n += 1
                        clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                        clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                        clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi
                        if s[i].shape[0] < 2:
                            break

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar, str(timedelta(seconds=(time.time()-start_time)))

def pruning_by_variance(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, del_neuron=True, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        in_error = mean_squared_error(y_v, clf_reg.predict(X_v))

    l_c = clf_reg.layers_count
    num_of_coefs = clf_reg.get_nuber_of_coefs()
    del_w = 0 #liczba usuniętych wag
    del_n = 0 #liczba usuniętych neuronów

    tmp_var, tmp_mean = met.outs_of_single_weight_for_variance(clf_reg, X_t)

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
        if np.all(np.isnan(tmp_val)):
            break
        tmp = np.nanargmin(np.abs(np.array(tmp_val)))
        tmp_var[tmp][tmp_ind[tmp]] = np.nan
        clf_reg.intercepts_[tmp][tmp_ind[tmp][1]] += tmp_mean[tmp][tmp_ind[tmp]]
        clf_reg.coefs_[tmp][tmp_ind[tmp]] = 0
        
        if del_neuron:
            for i in range(1,l_c): #sprawdzenie, czy usunąć neuron, gdy jego wszystkie wyjścia zostały przycięte; wagi między atrybirami, a pierwsza warstwą ukrytą są pomijane
                if tmp_var[i].shape[0] > 1: #czy w warstwie są przynajmniej 2 neurony
                    sums = np.nansum(tmp_var[i], axis=1)
                    for ind in (np.where(sums==0)[0])[::-1]:
                        tmp_var[i] = np.delete(tmp_var[i], ind, 0)
                        tmp_var[i-1] = np.delete(tmp_var[i-1], ind, 1)
                        tmp_mean[i] = np.delete(tmp_mean[i], ind, 0)
                        tmp_mean[i-1] = np.delete(tmp_mean[i-1], ind, 1)
                        del_w += np.sum(clf_reg.coefs_[i-1][:,ind] != 0) #usunięte wagi z kolumny warstwy poprzedzającej (potrzebne do zakończenia głównego while)
                        del_n += 1
                        clf_reg.coefs_[i] = np.delete(clf_reg.coefs_[i], ind, 0)
                        clf_reg.coefs_[i-1] = np.delete(clf_reg.coefs_[i-1], ind, 1)
                        clf_reg.intercepts_[i-1] = np.delete(clf_reg.intercepts_[i-1], ind, 0) #usunięcie bisu odpowiadającego usuwanemu neuronowi
                        if tmp_var[i].shape[0] < 2:
                            break

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return [del_w, del_n], miar, str(timedelta(seconds=(time.time()-start_time)))


def FBI_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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

        tmp = np.nanargmin(tmp_val) #indeks (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #indeks neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar, str(timedelta(seconds=(time.time()-start_time)))

def APERT_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
                        Sj[j] = met.APER(y_t, y_pred) - met.APER(y_t, y_pred0)
                    else:
                        Sj[j] = mean_squared_error(y_t, y_pred) - mean_squared_error(y_t, y_pred0) #dla regresji miara APER zastępiona błędem średniokwadratowym
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #indeks (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #indeks neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar, str(timedelta(seconds=(time.time()-start_time)))

def APERTP_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania; dla regresji działa identycznie, jak APERT
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
                        Sj[j] = met.APERP(y_t, y_pred) - met.APERP(y_t, y_pred0)
                    else:
                        Sj[j] = mean_squared_error(y_t, y_pred) - mean_squared_error(y_t, y_pred0) #dla regresji miara APERP zastępiona błędem średniokwadratowym
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #indeks (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #indeks neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar, str(timedelta(seconds=(time.time()-start_time)))

def PD_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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

        tmp = np.nanargmin(tmp_val) #indeks (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #indeks neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar, str(timedelta(seconds=(time.time()-start_time)))

def PEB_pruning(clf_reg, lost, X_t, y_t, X_v=None, y_v=None, refit=True, ep=20): #lost - maksymalna procentowa utrata dokładności podczas przycinania
    start_time = time.time()
    if clf_reg.coefs_[-1].shape[1] == 1:
        if_clf = False
    else:
        if_clf = True

    if X_v is None or y_v is None:
        X_v = X_t.copy()
        y_v = y_t.copy()

    if if_clf:
        in_acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
                Sj = np.mean(np.abs(np.sum(clf_reg.coefs_[i-1], axis=0)*(activ[i-1])), axis=0)
                tmp_ind[i] = np.argmin(Sj)
                tmp_val[i] = Sj[tmp_ind[i]]

        tmp = np.nanargmin(tmp_val) #indeks (+1) warstwy ukrytej, z której neuron ma zostać usunięty
        ind = tmp_ind[tmp] #indeks neuronu, który ma zostać usunięty

        clf_reg.coefs_[tmp] = np.delete(clf_reg.coefs_[tmp], ind, 0)
        clf_reg.coefs_[tmp-1] = np.delete(clf_reg.coefs_[tmp-1], ind, 1)
        clf_reg.intercepts_[tmp-1] = np.delete(clf_reg.intercepts_[tmp-1], ind, 0)

        if if_clf:
            acc = f1_score(y_v, clf_reg.predict(X_v), average='macro')
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
        miar = f1_score(y_v, clf_reg.predict(X_v), average='macro')
    else:
        miar = mean_squared_error(y_v, clf_reg.predict(X_v))
    if refit:
        clf_reg.refit(X_t, y_t, X_v, y_v, ep)
    return del_n, miar, str(timedelta(seconds=(time.time()-start_time)))
