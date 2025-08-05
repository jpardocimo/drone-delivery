"""
Módulo core.py
Implementa o Algoritmo Genético para o problema de entrega com drones.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
import random

# Tipos
Individuo = List[List[int]]  # Lista de duas sublistas, cada uma com índices de clientes da sublista é uma rota (gene), que juntas formam o individuo (genoma)
Populacao = List[Individuo] 


#-----------------------------
# Executa o Algoritmo Genético - FUNÇÃO PRINCIPAL DO GA
#-----------------------------
def algoritimo_genetico(
    clientes_coordenadas: np.ndarray,
    tamanho_populacao: int = 50,
    numero_de_geracoes: int = 100,
    limite_bateria: float = 4200.0,
    taxa_de_mutacao: float = 0.02,
    taxa_de_crossover: float = 0.8,
    tamanho_do_torneio: int = 3,
    percentual_de_elitismo: int = 10,
    parada_antecipada: bool = True,
    limite_da_parada_antecipada: int = 30,
    callback_on_generation: Optional[Callable[[int, Individuo, float], None]] = None
) -> Dict:
    """
    Executa o Algoritmo Genético e retorna o melhor resultado encontrado.
    Args:
        clientes_coordenadas: coordenadas dos clientes
        tamanho_populacao: tamanho da população
        numero_de_geracoes: numero de gerações
        limite_bateria: limite de bateria dos drones
        taxa_de_mutacao: taxa de mutação
        taxa_de_crossover: taxa de crossover
        tamanho_do_torneio: tamanho do torneio
        percentual_de_elitismo: percentual de elitismo
        parada_antecipada: se True, interrompe se não houver melhoria por limite_da_parada_antecipada gerações
        limite_da_parada_antecipada: número de gerações sem melhoria antes de parar (default: 30)
        callback_on_generation: função que será chamada a cada geração
    Returns:
        dict com as chaves: 'best_genotype', 'best_fitness', 'history', 'stopped_early' (lista de melhores fitness por geração)
    """
   
    #Inicialização da população
    numero_de_clientes = clientes_coordenadas.shape[0] #armazena a quantidade de clientes que precisam ser visitados
    populacao = inicializa_populacao(numero_de_clientes, tamanho_populacao) #cria a população inicial
    
    #------------------------------------------------
    # avalia a aptidão dos individuos da população
    #------------------------------------------------
    #calcula o fitness de cada individuo e armazena em uma lista (fitness da populacao)
    fitness_da_populacao = [calcula_fitness(individuo, clientes_coordenadas, limite_bateria) for individuo in populacao] 
    indice_do_melhor_individuo = int(np.argmin([m for m in fitness_da_populacao])) #indice do melhor individuo da população inicial
    melhor_individuo = [individuo.copy() for individuo in populacao[indice_do_melhor_individuo]] #armazena o melhor individuo
    melhor_fitness = fitness_da_populacao[indice_do_melhor_individuo] # só fitness do melhor individuo da população inicial
        
    # Variáveis para early stopping
    contador_de_melhoria = 0
    stopped_early = False

    #------------------------------------------------
    # Loop das gerações 
    #------------------------------------------------
    for geracao in range(numero_de_geracoes):
        nova_populacao = []

        # Converte o percentual em fração e calcula quantos indivíduos representam essa fração da população.
        # aplica o floor para arredondar para baixo e garantir que o numero de individuos seja inteiro
        num_elite = int(tamanho_populacao * percentual_de_elitismo / 100)

        # Ordenar população por fitness (melhores primeiro)
        indices_ordenados = np.argsort([fit for fit in fitness_da_populacao])

        # Seleciona os melhores individuos da população para elitismo
        elite_population = [populacao[i] for i in indices_ordenados[:num_elite]]

        # Adiciona os melhores individuos do elitismo na nova população
        nova_populacao = elite_population.copy()  # Preservar elite

        # Loop para seleção, crossover e mutação e gerar nova população
        while len(nova_populacao) < tamanho_populacao:
            # Seleção - Torneio
            pai1 = selecao(populacao, fitness_da_populacao, tamanho_do_torneio)
            pai2 = selecao(populacao, fitness_da_populacao, tamanho_do_torneio)
            # Crossover - OX (Ordered Crossover)
            if random.random() < taxa_de_crossover: #se a taxa de crossover for menor que um numero aleatorio, faz o crossover
                filho1, filho2 = order_crossover(pai1, pai2)
            else:
                filho1, filho2 = [rota.copy() for rota in pai1], [route.copy() for route in pai2]
            # Mutação - Swap de clientes dentro ou entre rotas
            filho1 = mutacao(filho1, taxa_de_mutacao)
            filho2 = mutacao(filho2, taxa_de_mutacao)
            nova_populacao.extend([filho1, filho2])
            
        # Ajusta tamanho da população
        populacao = nova_populacao[:tamanho_populacao]
        fitness_da_populacao = [calcula_fitness(individuo, clientes_coordenadas, limite_bateria) for individuo in populacao]
        indice_do_melhor_individuo_geracao = int(np.argmin([fit for fit in fitness_da_populacao]))
        melhor_fitness_geracao = fitness_da_populacao[indice_do_melhor_individuo_geracao]
        melhor_individuo_geracao = [rota.copy() for rota in populacao[indice_do_melhor_individuo_geracao]]
        
        # Atualiza o melhor individuo e o melhor fitness se o fitness da 
        # geração atual for menor que o melhor fitness encontrado até o momento
        if melhor_fitness_geracao < melhor_fitness:
            melhor_fitness = melhor_fitness_geracao
            melhor_individuo = [rota.copy() for rota in populacao[indice_do_melhor_individuo_geracao]]
            contador_de_melhoria = 0  # Reset contador
        else:
            contador_de_melhoria += 1
            
        # Early stopping check
        if parada_antecipada and contador_de_melhoria >= limite_da_parada_antecipada:
            stopped_early = True
            # Ainda chama o callback para mostrar a última geração
            if callback_on_generation is not None:
                callback_on_generation(geracao, melhor_individuo_geracao, melhor_fitness_geracao)
            break

        # Chamada do callback para animação
        if callback_on_generation is not None:
            callback_on_generation(geracao, melhor_individuo_geracao, melhor_fitness_geracao) 
    return {
        'melhor_individuo': melhor_individuo,
        'melhor_fitness': melhor_fitness,
        'stopped_early': stopped_early,
        'geracao_final': geracao + 1 if stopped_early else numero_de_geracoes
    }

#-----------------------------
# Cria a população inicial 
#-----------------------------
def inicializa_populacao(n_clientes: int, tam_pop: int) -> Populacao:
    """
    Gera `tam_pop` indivíduos.
    O tamanho da rota do Drone 1 (k) é sorteado uniformemente entre 0 e n_clientes.
    """
    clientes_base = list(range(n_clientes))
    populacao: Populacao = []

    for _ in range(tam_pop):
        permuta = clientes_base.copy() 
        random.shuffle(permuta)
        #permuta os clientes aleatoriamente entre os drones
        k = random.randint(0, n_clientes)   # ponto de corte aleatório
        drone1 = permuta[:k]
        drone2 = permuta[k:]
        populacao.append((drone1, drone2))

    return populacao

#-----------------------------
# Caucula o fitness de um individuo
#-----------------------------
def calcula_fitness(
    individuo,
    coords: np.ndarray,
    battery_limit: float,
    penalty_factor: float = 100.0
) -> Tuple[float, float]:
    """
    Retorna duas métricas:
    1. soma_dist - distância total percorrida pelos dois drones (já com penalidade de bateria)
    2. desequilíbrio -  diferença absoluta entre as duas distâncias
    """
    d1 = calcula_distancia_euclidiana_rota(individuo[0], coords)
    d2 = calcula_distancia_euclidiana_rota(individuo[1], coords)

    # Penalização de bateria - se a distancia percorrida for maior que o limite de bateria, 
    # penaliza com uma multa de 100 vezes a distancia excedente
    penalty = 0.0
    for d in (d1, d2):
        if d > battery_limit:
            penalty += (d - battery_limit) * penalty_factor

    soma_dist = d1 + d2 + penalty

    return soma_dist

#-----------------------------
# Seleção por torneio
#-----------------------------
def selecao(populacao: Populacao, array_de_fitness: List[float], tamanho_do_torneio: int = 3) -> Individuo:
    """
    Seleção por torneio: retorna um indivíduo da população.
    """
    # faz o torneio com um numero de individuos aleatorios da populacao
    selecionados = random.sample(list(enumerate(array_de_fitness)), tamanho_do_torneio)
    # retorna o indice do individuo com o menor fitness
    melhor_individuo = min(selecionados, key=lambda x: x[1])[0]
    # retorna o individuo com o menor fitness
    return [rota.copy() for rota in populacao[melhor_individuo]]

#-----------------------------
# Método de crossover ordenado (OX)
#-----------------------------
def order_crossover(pai1: Individuo, pai2: Individuo) -> Tuple[Individuo, Individuo]:
    """
    Crossover ordenado (OX) adaptado para duas rotas.
    """
    # Flatten ambos os pais
    pai1_flat = individuo_flat(pai1)
    pai2_flat = individuo_flat(pai2)
    tamanho_pai1_flat = len(pai1_flat)

    # Dois pontos de corte aleatórios para o ordered crossover
    ponto_a, ponto_b = sorted(random.sample(range(tamanho_pai1_flat), 2))

    # Filho 1
    filho1_meio = pai1_flat[ponto_a:ponto_b]
    filho1_resto = [x for x in pai2_flat if x not in filho1_meio]
    filho1_flat = filho1_resto[:ponto_a] + filho1_meio + filho1_resto[ponto_a:]

    # Filho 2
    filho2_meio = pai2_flat[ponto_a:ponto_b]
    filho2_resto = [x for x in pai1_flat if x not in filho2_meio]
    filho2_flat = filho2_resto[:ponto_a] + filho2_meio + filho2_resto[ponto_a:]

    # Divisão dos filhos: manter a mesma proporção dos pais, ou seja, 
    # para manter a quantidade de clientes de cada drone igual ao dos pais
    ponto_de_corte1 = len(pai1[0])
    ponto_de_corte2 = len(pai2[0])

    filho1 = individuo_split(filho1_flat, ponto_de_corte1)
    filho2 = individuo_split(filho2_flat, ponto_de_corte2)
    return filho1, filho2

#-----------------------------
# Mutação: troca de clientes dentro ou entre rotas 
# de acordo com a taxa de mutação
#-----------------------------
def mutacao(individuo: Individuo, taxa_de_mutacao: float) -> Individuo:
    """
    Mutação: troca de clientes dentro ou entre rotas.
    """
    novo_individuo = [rota.copy() for rota in individuo]
    # Troca dentro de cada rota
    for rota in novo_individuo:
        for i in range(len(rota)):
            if random.random() < taxa_de_mutacao and len(rota) > 1:
                j = random.randint(0, len(rota) - 1)
                if i != j:
                    rota[i], rota[j] = rota[j], rota[i]

    # Troca entre rotas
    if random.random() < taxa_de_mutacao and all(len(r) > 0 for r in novo_individuo):
        i = random.randint(0, len(novo_individuo[0]) - 1)
        j = random.randint(0, len(novo_individuo[1]) - 1)
        novo_individuo[0][i], novo_individuo[1][j] = novo_individuo[1][j], novo_individuo[0][i]
    return novo_individuo



####### FUNÇÕES AUXILIARES #######
#-----------------------------
# individuo_flat:concatena as rotas de dois drones em uma única lista 
# (junta as todas as rotas de um individuo em uma única lista, desconsiderando os drones)
#-----------------------------
def individuo_flat(individuo: Individuo) -> List[int]:
    """
    Concatena as rotas dos drones em uma única lista.
    a fim de facilitar o crossover e mutação
    """
    return individuo[0] + individuo[1]

#-----------------------------
# individuo_split: separa as rotas de uma lista flat em duas rotas (uma para cada drone) 
# de acordo com o ponto de corte 
# (separa rotas de uma única lista para colocalas separadas por drone novamente para formar um individuo)
#-----------------------------
def individuo_split(lista_rotas_flat: List[int], ponto_de_corte: int) -> Individuo:
    """
    Divide uma lista plana em duas rotas, dado o ponto de corte.
    """
    return [lista_rotas_flat[:ponto_de_corte], lista_rotas_flat[ponto_de_corte:]]

#-----------------------------
# Caucula a distancia euclidiana de uma rota para usar na função de fitness
#-----------------------------
def calcula_distancia_euclidiana_rota(route: List[int], coordenadas_clientes: np.ndarray) -> float:
    """
    Calcula a distância total de uma rota (lanchonete -> clientes -> lanchonete).
    """
    if not route:
        return 0.0
    total = 0.0
    # Da lanchonete ao primeiro cliente
    prev = np.array([0, 0])
    for idx in route:
        curr = coordenadas_clientes[idx]
        total += np.linalg.norm(curr - prev)
        prev = curr
    # Do último cliente de volta à lanchonete
    total += np.linalg.norm(prev - np.array([0, 0]))
    return total

