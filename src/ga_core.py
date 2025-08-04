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
        perm = clientes_base.copy()
        random.shuffle(perm)

        k = random.randint(0, n_clientes)   # ponto de corte aleatório
        drone1 = perm[:k]
        drone2 = perm[k:]

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

    # Penalização de bateria
    penalty = 0.0
    for d in (d1, d2):
        if d > battery_limit:
            penalty += (d - battery_limit) * penalty_factor

    soma_dist = d1 + d2 + penalty
    desequilibrio = abs(d1 - d2)
    return soma_dist, desequilibrio

#-----------------------------
# Seleção por torneio
#-----------------------------
def selecao(population: Populacao, fitnesses: List[float], tournament_size: int = 3) -> Individuo:
    """
    Seleção por torneio: retorna um indivíduo da população.
    """
    selected = random.sample(list(enumerate(fitnesses)), tournament_size)
    best_idx = min(selected, key=lambda x: x[1])[0]
    return [route.copy() for route in population[best_idx]]

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

    # Divisão dos filhos: manter proporção dos pais
    ponto_de_corte1 = len(pai1[0])
    ponto_de_corte2 = len(pai2[0])

    filho1 = individuo_split(filho1_flat, ponto_de_corte1)
    filho2 = individuo_split(filho2_flat, ponto_de_corte2)
    return filho1, filho2

#-----------------------------
# Mutação: troca de clientes dentro ou entre rotas 
# de acordo com a taxa de mutação
#-----------------------------
def mutate(individuo: Individuo, taxa_de_mutacao: float) -> Individuo:
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


#-----------------------------
# Executa o Algoritmo Genético
#-----------------------------
def algoritimo_genetico(
    clientes_coordenadas: np.ndarray,
    tamanho_populacao: int = 50,
    numero_de_geracoes: int = 100,
    limite_bateria: float = 1500.0,
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
        early_stopping: se True, interrompe se não houver melhoria por early_stopping_patience gerações
        early_stopping_patience: número de gerações sem melhoria antes de parar (default: 30)
    Returns:
        dict com as chaves: 'best_genotype', 'best_fitness', 'history', 'stopped_early' (lista de melhores fitness por geração)
    """
   
    numero_de_clientes = clientes_coordenadas.shape[0] #armazena a quantidade de clientes que precisam ser visitados
    populacao = inicializa_populacao(numero_de_clientes, tamanho_populacao) #cria a população inicial
    fitness_da_populacao = [calcula_fitness(individuo, clientes_coordenadas, limite_bateria) for individuo in populacao] #calcula o fitness de cada individuo e armazena em uma lista (fitness da populacao)
    indice_do_melhor_individuo = int(np.argmin([m[0] for m in fitness_da_populacao])) # menor soma_dist
    melhor_individuo = [individuo.copy() for individuo in populacao[indice_do_melhor_individuo]] #armazena o melhor individuo
    best_fitness = fitness_da_populacao[indice_do_melhor_individuo][0] # só a soma_dist
    history = [best_fitness]
    
    # Variáveis para early stopping
    no_improvement_count = 0
    stopped_early = False

    for gen in range(numero_de_geracoes):
        new_population = []
        num_elite = int(tamanho_populacao * percentual_de_elitismo / 100)
        # Ordenar população por fitness (melhores primeiro)
        sorted_indices = np.argsort([m[0] for m in fitness_da_populacao])
        elite_population = [populacao[i] for i in sorted_indices[:num_elite]]
        new_population = elite_population.copy()  # Preservar elite


        while len(new_population) < tamanho_populacao:
            # Seleção
            parent1 = selecao(populacao, fitness_da_populacao, tamanho_do_torneio)
            parent2 = selecao(populacao, fitness_da_populacao, tamanho_do_torneio)
            # Crossover
            if random.random() < taxa_de_crossover:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = [route.copy() for route in parent1], [route.copy() for route in parent2]
            # Mutação
            child1 = mutate(child1, taxa_de_mutacao)
            child2 = mutate(child2, taxa_de_mutacao)
            new_population.extend([child1, child2])
        # Ajusta tamanho
        populacao = new_population[:tamanho_populacao]
        fitness_da_populacao = [calcula_fitness(ind, clientes_coordenadas, limite_bateria) for ind in populacao]
        gen_best_idx = int(np.argmin([m[0] for m in fitness_da_populacao]))
        gen_best_fitness = fitness_da_populacao[gen_best_idx][0]
        gen_best_genotype = [route.copy() for route in populacao[gen_best_idx]]
        
        # Verifica se houve melhoria
        improved = False
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            melhor_individuo = [route.copy() for route in populacao[gen_best_idx]]
            improved = True
            no_improvement_count = 0  # Reset contador
        else:
            no_improvement_count += 1
            
        history.append(gen_best_fitness)
        
        # Early stopping check
        if parada_antecipada and no_improvement_count >= limite_da_parada_antecipada:
            print(f"🛑 Early stopping ativado: sem melhoria por {limite_da_parada_antecipada} gerações (geração {gen+1})")
            stopped_early = True
            # Ainda chama o callback para mostrar a última geração
            if callback_on_generation is not None:
                callback_on_generation(gen, gen_best_genotype, gen_best_fitness)
            break
            
        # Chamada do callback para animação
        if callback_on_generation is not None:
            callback_on_generation(gen, gen_best_genotype, gen_best_fitness) 
    return {
        'best_genotype': melhor_individuo,
        'best_fitness': best_fitness,
        'history': history,
        'stopped_early': stopped_early,
        'final_generation': gen + 1 if stopped_early else numero_de_geracoes
    }


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

#-----------------------------
# usada na selecao_pareto para desempate se a domina b segundo Pareto
#-----------------------------
def domina(a, b):
    """Retorna True se a domina b segundo Pareto (minimização)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])
from typing import Tuple

Metric = Tuple[float, float]

def domina(a: Metric, b: Metric) -> bool:
    """
    Verifica dominância de Pareto (minimização) em duas métricas.

    A domina B se:
      – não é pior em nenhuma métrica  e
      – é estritamente melhor em pelo menos uma.
    """
    dist_a, desequil_a = a
    dist_b, desequil_b = b

    nao_pior = (dist_a <= dist_b) and (desequil_a <= desequil_b)
    melhor_em_alguma = (dist_a < dist_b) or (desequil_a < desequilibrio_b)

    return nao_pior and melhor_em_alguma



#-----------------------------
# Seleção por torneio Pareto
#-----------------------------
def selecao_pareto(pop, metricas, k=3):
    """Torneio Pareto + desempate pela menor soma_dist."""
    participantes = random.sample(list(enumerate(metricas)), k)
    vencedor = participantes[0][0]
    for idx, m in participantes[1:]:
        if domina(metricas[idx], metricas[vencedor]):
            vencedor = idx
        elif not domina(metricas[vencedor], metricas[idx]):
            # Ninguém domina ninguém → menor soma_dist decide
            if m[0] < metricas[vencedor][0]:
                vencedor = idx
    return [r.copy() for r in pop[vencedor]]
# ---------------------------------------------------------------