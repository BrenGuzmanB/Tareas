"""
Created on Fri Nov  3 11:51:19 2023

@author: Bren Guzmán
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Paso 1: Cálculo de distancias
def calculate_distance_matrix(X):
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            distance = np.linalg.norm(X[i] - X[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

# Paso 2: Proyección inicial
iris = load_iris()
X = iris.data
distance_matrix = calculate_distance_matrix(X)

# Elegir una proyección inicial (en este caso, aleatoria)
np.random.seed(3)
X_mapped = np.random.rand(X.shape[0], 2)

# Visualización inicial
plt.scatter(X_mapped[:, 0], X_mapped[:, 1], c=iris.target)
plt.title("Iteración 0")
plt.show()



# Paso 4 y 5: Optimización y actualización de posiciones
max_iterations = 50
learning_rate = 0.1
convergence_threshold = 0.05  # Umbral de convergencia

prev_error = float("inf")

for iteration in range(1, max_iterations + 1):
    # Cálculo de distancias en la proyección
    mapped_distance_matrix = calculate_distance_matrix(X_mapped)

    gradient = np.zeros((X.shape[0], 2))

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i != j:
                d_real = distance_matrix[i, j]
                d_mapped = mapped_distance_matrix[i, j]

                if d_real != 0 and d_mapped != 0 and d_real != d_mapped:
                    delta = (d_real - d_mapped) / (d_mapped * (1 - d_mapped / d_real))
                    gradient[i] += (X_mapped[i] - X_mapped[j]) * delta

    # Actualizar las posiciones en la proyección
    X_mapped -= learning_rate * gradient

    # Calcular el error y verificar convergencia
    #error = (1/sum(distance_matrix))*(np.sum((distance_matrix - mapped_distance_matrix) ** 2/distance_matrix))
    error = np.sum((distance_matrix - mapped_distance_matrix) ** 2) / np.sum(distance_matrix)

    if abs(prev_error - error) < convergence_threshold:
        print(f"Convergencia alcanzada en la iteración {iteration}")
        break

    prev_error = error
    
    # Visualización en cada iteración
    plt.scatter(X_mapped[:, 0], X_mapped[:, 1], c=iris.target)
    
    plt.title(f"Iteración {iteration}")
    plt.show()

if iteration == max_iterations:
    print(f"No se alcanzó la convergencia después de {max_iterations} iteraciones.")
