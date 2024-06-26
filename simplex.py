import numpy as np
import pandas as pd

# Cantidad de variables
numVar = int(input("Ingrese la cantidad de variables: "))

# Cantidad de restricciones
numRest = int(input("Ingrese la cantidad de restricciones: "))

# Tipo de problema
tipo = int(input(
    "¿Es un problema de maximización o minimización?\n Maximizar [1] \n Minimizar[2]: "
))

# Creación de la función objetivo
for i in range(numVar + 1):
    if i == 0:
        matZ = np.array([1])
    else:
        print(
            "Ingrese el coeficiente de la variable x"
            + str(i)
            + " en la función objetivo: "
        )
        coef = float(input())
        # Agregar coeficientes a la matriz Z
        matZ = np.append(matZ, coef)

# Creación de las restriccion
# Inicializa matA con ceros
matA = np.zeros(
    (numRest, numVar + numRest + 2)
)  # Se añade 2 para la columna Z y el valor RHS

 #En caso de que la restricción sea de igualdad
restExt = np.array([])
varExt = np.array([])
numRestExt = numRest

for i in range(numRest):
    matR = np.zeros(numVar + numRest + 2)  # Se corrige el tamaño de matR
    if i == 0:
        matR[0] = 1
    else:
        matR[0] = 0

    for j in range(numVar):
        print(
            "Ingrese el coeficiente de la variable x"
            + str(j + 1)
            + " en la restricción "
            + str(i + 1)
            + ": "
        )
        coef = float(input())
        matR[j + 1] = coef

    print("Ingrese el tipo de restricción: \n Menor o igual [1] \n Mayor o igual [2] \n Igual [3]: ")
    tipoRest = int(input())
    print("Ingrese el valor del lado derecho de la restricción " + str(i + 1) + ": ")
    ladoDer = float(input())
    matR[-1] = ladoDer  # Valor RHS

    matR[numVar + i + 1] = 1  # Variable de holgura

    matA[i, :] = matR  # Asigna matR a la fila correspondiente en matA
    
    if tipo == 1 and tipoRest == 2:
        matA[i, 1:numVar + 1] = -matA[i, 1:numVar + 1]
        # print("Volteando los coeficientes")
    
    if tipo == 2 and tipoRest == 1:
        matA[i, 1:numVar + 1] = -matA[i, 1:numVar + 1]
        matA[i, -1] = -matA[i, -1]

        # print("Volteando los coeficientes")
    
    # if tipoRest == 3:
    #     nuevaHolgura = "h" + str(numRestExt + 1)
    #     varExt = np.append(varExt, nuevaHolgura)
    #     numRestExt += 1
        
      
# print("Variables extra", varExt)        
    # print("matA: \n", matA)

# Agregar holguras a la función objetivo
for i in range(numRest + 1):
    matZ = np.append(matZ, 0)

# Crear encabezados de las variables

for i in range(numVar + 1):
    if i == 0:
        var = np.array(["Z"])
    else:
        var = np.append(var, "x" + str(i))

for i in range(numRest):
    var = np.append(var, "h" + str(i + 1))

var = np.append(var, "Const")


# Volver negativos los coeficientes de la función objetivo
matZ = -matZ
matZ[0] = -matZ[0]
matZ[-1] = -matZ[-1]


def simplex(matA, matZ):
    # Se obtiene el número de filas y columnas de la matriz A
    numFilas, numCols = matA.shape

    # Se obtiene el número de variables básicas
    numVarBas = numFilas

    # Se obtiene el número de variables no básicas
    numVarNoBas = numCols - numVarBas

    # Valor más negativo de la fila Z
    valNeg = np.min(matZ)
    # Indice de la columna pivote
    colPiv = np.where(matZ == valNeg)[0][0]

    # Se obtiene el renglón pivote. Busca el menor coeficiente de const/coeficiente de la columna pivote
    minVal = np.inf
    filPiv = 0
    for i in range(numFilas):
        if matA[i, colPiv] > 0:
            val = matA[i, -1] / matA[i, colPiv]
            if val < minVal:
                minVal = val
                filPiv = i

    # Se obtiene el elemento pivote
    elemPiv = matA[filPiv, colPiv]
    print("Elemento pivote: ", elemPiv)
    print("\n")

    # Volvemos 1 al elemento pivote
    matA[filPiv, :] = matA[filPiv, :] / elemPiv

    # Hacer ceros los elementos de la columna pivote
    for i in range(numFilas):
        if i != filPiv:
            matA[i, :] = matA[i, :] - matA[i, colPiv] * matA[filPiv, :]

    # Hacer cero el elemento de la fila Z
    matZ = matZ - matZ[colPiv] * matA[filPiv, :]

    # Comprobar que no haya valores negativos en la fila Z
    valNeg = np.min(matZ)
    if valNeg < 0:
        fin = False
    else:
        fin = True                    

    return matA, matZ, fin

def obtener_valores_finales(matA, matZ, var_names):
    # Inicializar los valores finales de todas las variables en 0
    valores_finales = dict.fromkeys(var_names, 0.0)
    
    # El valor final de Z es el último elemento de matZ
    valores_finales['Z'] = matZ[-1]
    
    numFilas, numCols = matA.shape
    
    # Recorrer cada columna para identificar las variables básicas y sus valores
    for i, var in enumerate(var_names[:-1]):  # Excluir la columna de constantes de var_names
        col = matA[:, i]  # Extraer la columna actual
        
        # Si en la columna hay exactamente un 1 y el resto son 0s, se considera una variable básica
        if np.count_nonzero(col == 1) == 1 and np.count_nonzero(col == 0) == numFilas - 1:
            fila = np.where(col == 1)[0][0]  # Encontrar la fila del 1
            valores_finales[var] = matA[fila, -1]  # Asignar el valor de la constante de esa fila
    
    # Eliminar la tupla "Const"
    if 'Const' in valores_finales:  
        del valores_finales['Const']
    
    return valores_finales


def obtenerDual(matA, matZ):

    # El ultimo elemento de cada fila en MatA será el nuevo valor de cada variable en la función objetivo
    dual = matA[:, -1]
    nuevMatZ = np.zeros(len(dual) + 1)
    # Primer elemento de la matriz Z es 1

    nuevMatZ[0] = 1
    nuevVar = np.array(["Z"])
    nuevMatA = np.empty(0)

    # Agregar los valores del dual a la matriz Z
    for i in range(len(dual)):
        nuevMatZ[i + 1] = dual[i]
        nuevVar = np.append(nuevVar, "x" + str(i + 1))
    # Agregar los valores de holgura a la matriz Z
    for i in range(numVar):
        nuevMatZ = np.append(nuevMatZ, 0)
        nuevVar = np.append(nuevVar, "h" + str(i + 1))

    nuevVar = np.append(nuevVar, "Const")

    # Los valores de cada columna en matA serán los coeficientes de cada restricción
    for i in range(len(dual)):
        matA[i, -1] = 0
    # print(matA)

    intermediate_cols = matA[:, 1:-1].T
    nuevMatA = np.empty((0, numRest))
    for i in range(numVar):
        nuevMatA = np.vstack((nuevMatA, intermediate_cols[i]))
        # print("intermediate_cols[i]: ", intermediate_cols[i])

    for i in range(numVar):
        holgura = np.zeros(
            (nuevMatA.shape[0], 1)
        )  # Crea una columna de ceros para la holgura
        holgura[i, 0] = 1  # Agrega un 1 en la fila correspondiente para la holgura
        nuevMatA = np.hstack(
            [nuevMatA, holgura]
        )  # Agrega la columna de holgura a la matriz

    nuevMatA = np.insert(nuevMatA, 0, values=0, axis=1)
    nuevMatZ = np.append(nuevMatZ, 0)

    nuevasRest = matZ[1 : 1 + numVar]
    nuevasRest = -nuevasRest
    # print("nuevasRest: ", nuevasRest)

    nuevMatA = np.column_stack((nuevMatA, nuevasRest))

    # print("matA: ", matA)
    # print("intermediate_cols: ", intermediate_cols)
    # print("nuevMatA: ", nuevMatA)
    # print(matZ)
    nuevMatZ = -nuevMatZ
    nuevMatZ[0] = -nuevMatZ[0]
    nuevMatZ[-1] = -nuevMatZ[-1]
    # print("Las variables son: ", nuevVar)
    # print("La matriz Z es: ", nuevMatZ)
    # print("La matriz A es: \n", nuevMatA)

    return nuevMatA, nuevMatZ, nuevVar


if tipo == 1:
    fin = False
    print("Matriz inicial")
    df = pd.DataFrame([matZ] + matA.tolist(), columns=var)
    df.iloc[1:, 0] = (
        0  # Asegurándose de que todas las entradas bajo "Z" en restricciones sean 0
    )
    print(df)
    with pd.ExcelWriter('salida_final.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Inicial')

        # Se ejecuta el método simplex
        iteracion = 1
        while not fin:
            matA, matZ, fin = simplex(matA, matZ)
            print("Iteración ", iteracion)
            # Se crea un DataFrame con las matrices
            matZ[0] = 1.0  # Ajusta el valor de Z a 1
            df = pd.DataFrame([matZ] + matA.tolist(), columns=var)
            df.iloc[1:, 0] = (
                0  # Asegurándose de que todas las entradas bajo "Z" en restricciones sean 0
            )
            print(df)
            df.to_excel(writer, index=False, sheet_name=f'Iteración {iteracion}')
            iteracion += 1
        if fin:
            print("Solución óptima encontrada")
            matZ[0] = 1.0  # Ajusta el valor de Z a 1
            df = pd.DataFrame([matZ] + matA.tolist(), columns=var)
            df.iloc[1:, 0] = (
                0  # Asegurándose de que todas las entradas bajo "Z" en restricciones sean 0
            )
            print(df)
            
            valores_finales = obtener_valores_finales(matA, matZ, var)
            print("Valores finales: ", valores_finales)
            df.to_excel(writer, index=False, sheet_name='Solución')


if tipo == 2:
    # Se obtiene la solución dual
    newMatA, newMatZ, newVar = obtenerDual(matA, matZ)
    fin = False
    print("Matriz inicial")
    df = pd.DataFrame([newMatZ] + newMatA.tolist(), columns=newVar)
    df.iloc[1:, 0] = (
        0  # Asegurándose de que todas las entradas bajo "Z" en restricciones sean 0
    )
    print(df)
    with pd.ExcelWriter('salida_final.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Inicial')

        # Se ejecuta el método simplex
        iteracion = 1
        while not fin:
            newMatA, newMatZ, fin = simplex(newMatA, newMatZ)
            print("Iteración ", iteracion)
            # Se crea un DataFrame con las matrices
            matZ[0] = 1.0  # Ajusta el valor de Z a 1
            df = pd.DataFrame([newMatZ] + newMatA.tolist(), columns=newVar)
            df.iloc[1:, 0] = (
                0  # Asegurándose de que todas las entradas bajo "Z" en restricciones sean 0
            )
            print(df)
            df.to_excel(writer, index=False, sheet_name=f'Iteración {iteracion}')
            iteracion += 1
        if fin:
            print("\nSolución óptima encontrada")
            newMatZ[0] = 1.0  # Ajusta el valor de Z a 1
            df = pd.DataFrame([newMatZ] + newMatA.tolist(), columns=newVar)
            df.iloc[1:, 0] = (
                0  # Asegurándose de que todas las entradas bajo "Z" en restricciones sean 0
            )
            print(df)
            df.to_excel(writer, index=False, sheet_name='Solución')

