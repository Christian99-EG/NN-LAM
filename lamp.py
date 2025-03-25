import numpy as np

def vectores_A():
    num_rows = int(input("Número de filas: "))
    num_col = int(input("Número de columnas: "))
    
    matriz_A = []
    for i in range(num_rows):
        row = [int(input(f"Introduce un valor para el vector {i}: ")) for j in range(num_col)]
        matriz_A.append(row)
    
    return np.array(matriz_A)

def vectores_B():
    num_rows = int(input("Número de filas: "))
    num_col = int(input("Número de columnas: "))
    
    matriz_B = []
    for i in range(num_rows):
        row = [int(input(f"Introduce un valor para el vector {i+1}: ")) for j in range(num_col)]
        matriz_B.append(row)
    
    return np.array(matriz_B)

def calcula_pesos(va, vb):
    if va.shape[0] != vb.shape[0]:
        raise ValueError("Las listas de vectores A y B deben tener el mismo número de filas (muestras M).")
    
    A_transformada = 2 * va - 1
    B_transformada = 2 * vb - 1
    
    W = np.zeros((va.shape[1], vb.shape[1]))
    for i in range(va.shape[0]):
        W += np.outer(A_transformada[i], B_transformada[i])
    
    return W

def calcula_bias(va):
    return -0.5 * np.sum(va, axis=0)

def salida_lam(va, vw, vbias):
    new_va = np.shape(va)
    new_vw = np.shape(vw)

    return np.matmul(new_va, new_vw) + vbias 
#se cambio dot() por multuply

def validar_asociacion(A, B, W, bias):
    salida_calculada = salida_lam(A, W, bias)
    #bool_A = np.array()
    #return np.allclose(salida_calculada, B)
    return np.array_equal(np.sign(salida_calculada), np.sign(B))

print("Vectores de entrada A para la red LAM")
salida_A = vectores_A()
print("\nVectores de salida B de la entrada A")
salida_B = vectores_B()

W = calcula_pesos(salida_A, salida_B)
bias = calcula_bias(salida_A)
resultado = salida_lam(salida_A, W, bias)

print("Matriz de pesos W:")
print(W)
print("\nResultado de LAM:")
print(resultado)

longitud = len(salida_A)
if validar_asociacion(salida_A, salida_B, W, bias):
    print("Los vectores de entrada se asocian correctamente con los vectores de salida.")
    for i in range (longitud):
        print(f"vector A{i} asocia con el vector B{i}")
    print(np.sign(salida_A), np.sign(salida_B))
else:
    print("asociación de los vectores de entrada con los de salida.")
    for i in range (longitud):
            print(f"vector A{i} asocia con el vector B{i}")