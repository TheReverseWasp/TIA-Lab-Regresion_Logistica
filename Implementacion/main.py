from experimentos import * 

def main():
    file_names = ["../Datos/diabetes.csv", "../Datos/Enfermedad_Cardiaca.csv"]
    print("Seleccione el experimento a realizar: ")
    print("Experimento_1 Check Accuracy")
    print("Experimento_2 Check Cost")
    opcion = check_type(int)
    while opcion < 3 and opcion > 0:
        if opcion == 1:
            Experimento_1(file_names)
        else:
            Experimento_2(file_names)
        print("Datos Guardados en la ubicacion ../Resultados/Exp1 o Exp2")
        print("Seleccione el experimento a realizar: ")
        print("Experimento_1 Check Accuracy")
        print("Experimento_2 Check Cost")
        opcion = check_type(int)

if __name__ == "__main__":
    main()
