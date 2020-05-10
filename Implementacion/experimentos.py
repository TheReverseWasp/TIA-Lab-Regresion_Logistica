from my_lib import *

lr_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
lr_str_save_lst = ["0_01", "0_05", "0_1", "0_2", "0_3", "0_4"]
it_list = [500, 1000, 1500, 2000, 2500, 3000, 3500]
true_it_list = [501, 1001, 1501, 2001, 2501, 3001, 3501]
k = 3

def Experimento_1(file_names):
    fold_list = permutation(np.arange(k).tolist())
    dataset_number = 0
    for i in file_names:
        dataset_number += 1
        #try:
        np_arr = Leer_Datos(i)
        np_arr, __, __ = Normalizar_Datos(np_arr)
        dict_k_folds = Crear_k_folds(np_arr, k)
        perm = 0
        for p in fold_list:
            perm += 1
            print("permutacion " + str(perm) + ": " + str(p))
            join_data_train = join_arrs(dict_k_folds["k" + str(p[0])], dict_k_folds["k" + str(p[1])])
            for i in range(2, k - 1):
                join_data_train = join_arrs(join_data_train, dict_k_folds["k" + str(p[i])])
            X_train, Y_train = separador(join_data_train)
            X_test, Y_test = separador(dict_k_folds["k" + str(p[-1])])
            updated_l_accuracy_train = []
            updated_l_accuracy_test = []
            for lr in lr_list:
                theta = init_theta(X_train)
                theta, lista_costos, lista_accuracy, lista_thetas = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = lr)
                updated_l_accuracy_train.append(lista_accuracy)
                temp = []
                for theta_1 in lista_thetas:
                    temp.append(Calcular_Accuracy(X_test, theta_1, Y_test))
                updated_l_accuracy_test.append(temp)
            updated_l_accuracy_train = np.array(updated_l_accuracy_train).T.tolist()
            updated_l_accuracy_test = np.array(updated_l_accuracy_test).T.tolist()
            with open("../Resultados/Exp1/train/" + str(dataset_number) + "-train-p" + str(perm) + ".txt", "w") as file:
                file.write("permutacion: " + str(p) + "\n")
                file.write("filas: " + str(it_list) + "\n")
                file.write("columnas: " + str(lr_list) + "\n")
                file.write("\n\n" + str(updated_l_accuracy_train))
            with open("../Resultados/Exp1/test/" + str(dataset_number) + "-test-p" + str(perm) + ".txt", "w") as file:
                file.write("permutacion: " + str(p) + "\n")
                file.write("filas: " + str(it_list) + "\n")
                file.write("columnas: " + str(lr_list) + "\n")
                file.write("\n\n" + str(updated_l_accuracy_test))
        #end of exp1
        #except:
        #    print("Lectura de archivos Fallida")
    return False

def Experimento_2(file_names):
    fold_list = permutation(np.arange(k).tolist())
    dataset_number = 0
    for i in file_names:
        dataset_number += 1
        #try:
        np_arr = Leer_Datos(i)
        np_arr, __, __ = Normalizar_Datos(np_arr)
        dict_k_folds = Crear_k_folds(np_arr, k)
        perm = 0
        for p in fold_list:
            perm += 1
            print("permutacion " + str(perm) + ": " + str(p))
            join_data_train = join_arrs(dict_k_folds["k" + str(p[0])], dict_k_folds["k" + str(p[1])])
            for i in range(2, k - 1):
                join_data_train = join_arrs(join_data_train, dict_k_folds["k" + str(p[i])])
            X_train, Y_train = separador(join_data_train)
            X_test, Y_test = separador(dict_k_folds["k" + str(p[-1])])
            list_item = 0
            for lr in lr_list:
                theta = init_theta(X_train)
                theta, lista_costos, lista_accuracy, lista_thetas = Gradiente_Descendiente(X_train, theta, Y_train, learning_rate = lr, step = 50)
                path_to_save = "../Resultados/Exp2/" + str(dataset_number) + "-" + lr_str_save_lst[list_item] + "-p" + str(perm) + ".jpg"
                list_item += 1
                lista_costos_test = []
                for theta_item in lista_thetas:
                    lista_costos_test.append(Calcular_Funcion_Costo(X_test, theta_item, Y_test))
                Graficar_Costo_2(lista_costos, lista_costos_test, path_to_save)
        #except:
        #    print("Lectura de archivos Fallida")
    return False
