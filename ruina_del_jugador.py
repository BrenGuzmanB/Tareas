"""
Created on Tue Oct 31 11:38:43 2023

@author: Bren Guzmán
"""

import random

def jugar_moneda(capital, apuesta, probabilidad_ganar):
    if random.random() < probabilidad_ganar:
        return capital + apuesta
    else:
        return capital - apuesta

def simulacion_ruina_jugador(capital_inicial, apuesta, probabilidad_ganar, n_juegos):
    capital = capital_inicial
    resultados = [capital]

    for _ in range(n_juegos):
        if capital <= 0:
            break
        capital = jugar_moneda(capital, apuesta, probabilidad_ganar)
        resultados.append(capital)

    return resultados

capital_inicial = 1000  # Capital inicial del jugador
apuesta = 10           # Cantidad apostada en cada juego
probabilidad_ganar = 0.5  # Probabilidad de ganar en cada juego
n_juegos = 1000        # Número de juegos a simular

resultados = simulacion_ruina_jugador(capital_inicial, apuesta, probabilidad_ganar, n_juegos)

import matplotlib.pyplot as plt

plt.plot(resultados)
plt.xlabel("Número de juegos")
plt.ylabel("Capital del jugador")
plt.title("Simulación de la Ruina del Jugador")
plt.show()
