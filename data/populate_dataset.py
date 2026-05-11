import pandas as pd 
import numpy as np 

num_linhas = 500 

dados = []


for i in range(1, num_linhas + 1):

    nome = f"jogador_{i}"

    idade = np.random.randint(18,40)

    distancia_jogo = round(np.random.uniform(0,13),1)
    distancia_treino = round(np.random.uniform(),1)

    sprints_jogo =  np.random.randint(10, 65)
    sprints_treino = np.rando.randint(30,160)

    altura_salto = round(np.random.uniform(25, 60), 1)

    capacidade_aerobica = round(np.random.uniform(45, 100), 1)
    capacidade_anaerobica = round(np.random.uniform(45, 100), 1)

    # 0 = não teve lesão recente, 1 = teve lesão recente
    lesao_recente = np.random.choice([0, 1], p=[0.75, 0.25])

    # 0 = tem dores, 1 = não tem dores
    ausencia_dores = np.random.choice([0, 1], p=[0.30, 0.70])

    tempo_descanso_entre_jogos = np.random.randint(2, 9)


    score = 0

    if idade >= 30:
        score += 1

    if distancia_jogo > 10.5:
        score_risco += 1

    if distancia_treino > 35:
        score_risco += 1

    if sprints_jogo > 45:
        score_risco += 1

    if sprints_treino > 110:
        score_risco += 1

    if altura_salto < 35:
        score_risco += 1

    if capacidade_aerobica < 65:
        score_risco += 1

    if capacidade_anaerobica < 65:
        score_risco += 1

    if lesao_recente == 1:
        score_risco += 2

    if ausencia_dores == 0:
        score_risco += 2

    if tempo_descanso_entre_jogos <= 3:
        score_risco += 1

    if score <= 2:
        risco_lesao = 0
    elif score > 2 and score <= 5:
        risco_lesao = 1
    else:
        risco_lesao = 2

dados.append([
    nome, 
    idade,
    distancia_jogo,
    distancia_treino,
    altura_salto,
    capacidade_aerobica,
    capacidade_anaerobica,
    lesao_recente,
    ausencia_dores,
    tempo_descanso_entre_jogos,
    risco_lesao
])

df = pd.DataFrame(dados, columns=[
    "nome",
    "idade",
    "distancia_jogo",
    "distancia_treino",
    "sprints_jogo",
    "sprints_treino",
    "altura_salto",
    "capacidade_aerobica",
    "capacidade_anaerobica",
    "lesao_recente",
    "ausencia_dores",
    "tempo_descanso_entre_jogos",
    "risco_lesao"
])

df.to_csv("dataset1.csv")

