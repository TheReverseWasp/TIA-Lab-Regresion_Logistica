from complementos import *

eps = 1e-10

#1
def Leer_Datos(filename):
    df = pd.read_csv(filename, sep = "\t")
    np_arr = df.to_numpy()
    np_arr = np_arr.T
    temp = [np.ones(np_arr.shape[1]).tolist()]
    for i in np_arr:
        temp.append(i.tolist())
    answer = np.asarray(temp)
    return answer

#2
def Normalizar_Datos(np_arr):
    Media, Desviacion = desviacion_estandar_2(np_arr[1:-1])
    np_arr[1:-1] = (np_arr[1:-1] - Media) / Desviacion
    return np_arr, Media, Desviacion

def Normalizar_Datos_MD(np_arr, Media, Desviacion):
    np_arr[1:-1] = (np_arr[1:-1] - Media) / Desviacion
    return answer

#3
def Sigmoidal(X, theta):
    return 1/(1 + np.exp(-np.dot(theta.T, X) + eps) + eps)

#4
def Calcular_Funcion_Costo(X, theta, Y):
    return -np.sum(Y * np.log(Sigmoidal(X, theta) + eps) + (1 - Y) * np.log(1 - Sigmoidal(X, theta) + eps)) / Y.shape[1]

#5
def Calcular_Gradiente(X, theta, Y):
    return np.sum(np.dot(X, (Sigmoidal(X, theta) - Y).T), axis = 1, keepdims = True) / Y.shape[1]

#7
def Calcular_Accuracy(X, theta, Y):
    predicciones = (Sigmoidal(X, theta) >= .5).astype(int)
    comparacion = (predicciones == Y).astype(float)
    #print(comparacion)
    unique, counts = np.unique(comparacion, return_counts = True)
    dict_t = dict(zip(unique, counts))
    return (dict_t[1] / comparacion.shape[1])

#6
def Gradiente_Descendiente(X, theta, Y, iteraciones = 3501, learning_rate = 0.4, step = 500):
    lista_costos = []
    lista_accuracy = []
    lista_thetas = []
    for it in range(1, iteraciones):
        theta = theta - learning_rate * Calcular_Gradiente(X, theta, Y)
        if it % step == 0:
            lista_costos.append(Calcular_Funcion_Costo(X, theta, Y))
            lista_accuracy.append(Calcular_Accuracy(X, theta, Y))
            lista_thetas.append(theta)
    return theta, lista_costos, lista_accuracy, lista_thetas

#8
def Crear_k_folds(np_arr, k = 3): # Only works with y = 0 or 1
    unique, counts = np.unique(np_arr[-1], return_counts = True)
    np_arr = np_arr.T
    dict_unique = {}
    for i in unique:
        dict_unique[i] = []
    for i in np_arr:
        dict_unique[i[-1]].append(i.tolist())
    dict_answer = {}
    for i in range(k):
        dict_answer["k" + str(i)] = []
    for i in range(k - 1):
        for u, c in zip(unique, counts):
            for j in range(int(c / k) * i, int(c / k) * (i + 1)):
                dict_answer["k" + str(i)].append(dict_unique[u][j])
    for u, c in zip(unique, counts):
        for j in range(int(c / k) * (k - 1), c):
            dict_answer["k" + str(k - 1)].append(dict_unique[u][j])
    for i in range(k):
        temp = np.array(dict_answer["k" + str(i)])
        np.random.shuffle(temp)
        dict_answer["k" + str(i)] = temp.T
    return dict_answer
