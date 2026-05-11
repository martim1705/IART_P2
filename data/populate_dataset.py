import pandas as pd 
import numpy as np 
import os 

num_linhas = 500 

for i in range(1, num_linhas + 1):

    nome = f"jogador_{i}"

    idade = np.random.randint(18,40)

    distancia_jogo = np.random.uniform(0,13)
    distancia_treino = np.random.uniform()