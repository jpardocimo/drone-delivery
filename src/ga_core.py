"""
Módulo core.py
Implementa o Algoritmo Genético para o problema de entrega com drones.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
import random

# Tipos
Genotype = List[List[int]]  # Lista de duas sublistas, cada uma com índices de clientes
Population = List[Genotype]


def domina(a, b):
    """Retorna True se a domina b segundo Pareto (minimização)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

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

#-----------------------------
# Cria a população inicial 
#-----------------------------
def initialize_population(num_clients: int, population_size: int) -> Population:
    """
    Inicializa a população com permutações aleatórias dos clientes divididas de forma equilibrada entre dois drones.
    """
    population = []
    clients = list(range(num_clients))
    for _ in range(population_size):
        perm = clients.copy()
        random.shuffle(perm)
        # Divisão equilibrada
        split = num_clients // 2
        drone1 = perm[:split]
        drone2 = perm[split:]
        population.append([drone1, drone2])
    return population


def route_distance(route: List[int], clients_coords: np.ndarray) -> float:
    """
    Calcula a distância total de uma rota (lanchonete -> clientes -> lanchonete).
    """
    if not route:
        return 0.0
    total = 0.0
    # Da lanchonete ao primeiro cliente
    prev = np.array([0, 0])
    for idx in route:
        curr = clients_coords[idx]
        total += np.linalg.norm(curr - prev)
        prev = curr
    # Do último cliente de volta à lanchonete
    total += np.linalg.norm(prev - np.array([0, 0]))
    return total


def calcula_fitness(
    genotype,
    coords: np.ndarray,
    battery_limit: float,
    penalty_factor: float = 100.0
) -> Tuple[float, float]:
    """
    Retorna duas métricas:
    1. soma_dist  – distância total percorrida pelos dois drones (já com penalidade de bateria)
    2. desequilíbrio – diferença absoluta entre as duas distâncias
    """
    d1 = route_distance(genotype[0], coords)
    d2 = route_distance(genotype[1], coords)

    # Penalização de bateria
    penalty = 0.0
    for d in (d1, d2):
        if d > battery_limit:
            penalty += (d - battery_limit) * penalty_factor

    soma_dist = d1 + d2 + penalty
    desequilibrio = abs(d1 - d2)
    return soma_dist, desequilibrio

# def calcula_fitness(genotype: Genotype, clients_coords: np.ndarray, battery_limit: float) -> float:
#     """
#     Calcula o fitness (distância total percorrida pelos drones), penalizando se exceder o limite de bateria.
#     """
#     penalty_factor = 100.0
#     total = 0.0
#     for route in genotype:
#         dist = route_distance(route, clients_coords)
#         if dist > battery_limit:
#             dist *= penalty_factor
#         total += dist
#     return total


def selecao(population: Population, fitnesses: List[float], tournament_size: int = 3) -> Genotype:
    """
    Seleção por torneio: retorna um indivíduo da população.
    """
    selected = random.sample(list(enumerate(fitnesses)), tournament_size)
    best_idx = min(selected, key=lambda x: x[1])[0]
    return [route.copy() for route in population[best_idx]]


def flatten_genotype(genotype: Genotype) -> List[int]:
    """
    Concatena as rotas dos drones em uma única lista.
    """
    return genotype[0] + genotype[1]


def split_genotype(flat: List[int], split_point: int) -> Genotype:
    """
    Divide uma lista plana em duas rotas, dado o ponto de corte.
    """
    return [flat[:split_point], flat[split_point:]]


def order_crossover(parent1: Genotype, parent2: Genotype) -> Tuple[Genotype, Genotype]:
    """
    Crossover ordenado (OX) adaptado para duas rotas.
    """
    # Flatten ambos os pais
    flat1 = flatten_genotype(parent1)
    flat2 = flatten_genotype(parent2)
    n = len(flat1)
    # Pontos de corte
    a, b = sorted(random.sample(range(n), 2))
    # Filho 1
    child1_mid = flat1[a:b]
    child1_rest = [x for x in flat2 if x not in child1_mid]
    child1_flat = child1_rest[:a] + child1_mid + child1_rest[a:]
    # Filho 2
    child2_mid = flat2[a:b]
    child2_rest = [x for x in flat1 if x not in child2_mid]
    child2_flat = child2_rest[:a] + child2_mid + child2_rest[a:]
    # Divisão dos filhos: manter proporção dos pais
    split1 = len(parent1[0])
    split2 = len(parent2[0])
    child1 = split_genotype(child1_flat, split1)
    child2 = split_genotype(child2_flat, split2)
    return child1, child2


def mutate(genotype: Genotype, mutation_rate: float) -> Genotype:
    """
    Mutação: troca de clientes dentro ou entre rotas.
    """
    new_genotype = [route.copy() for route in genotype]
    # Troca dentro de cada rota
    for route in new_genotype:
        for i in range(len(route)):
            if random.random() < mutation_rate and len(route) > 1:
                j = random.randint(0, len(route) - 1)
                if i != j:
                    route[i], route[j] = route[j], route[i]
    # Troca entre rotas
    if random.random() < mutation_rate and all(len(r) > 0 for r in new_genotype):
        i = random.randint(0, len(new_genotype[0]) - 1)
        j = random.randint(0, len(new_genotype[1]) - 1)
        new_genotype[0][i], new_genotype[1][j] = new_genotype[1][j], new_genotype[0][i]
    return new_genotype


def genetic_algorithm(
    clients_coords: np.ndarray,
    population_size: int = 50,
    generations: int = 100,
    battery_limit: float = 1500.0,
    mutation_rate: float = 0.02,
    crossover_rate: float = 0.8,
    tournament_size: int = 3,
    on_generation: Optional[Callable[[int, Genotype, float], None]] = None
) -> Dict:
    """
    Executa o Algoritmo Genético e retorna o melhor resultado encontrado.
    Returns:
        dict com as chaves: 'best_genotype', 'best_fitness', 'history' (lista de melhores fitness por geração)
    """
    num_clients = clients_coords.shape[0]
    population = initialize_population(num_clients, population_size)
    #fitnesses = [calcula_fitness(ind, clients_coords, battery_limit) for ind in population]
    # best_idx = int(np.argmin(fitnesses))
    # best_fitness = fitnesses[best_idx]
    metricas = [calcula_fitness(ind, clients_coords, battery_limit) for ind in population]
    best_idx = int(np.argmin([m[0] for m in metricas]))     # menor soma_dist
    best_genotype = [route.copy() for route in population[best_idx]]
    best_fitness = metricas[best_idx][0]                    # só a soma
    history = [best_fitness]

    for gen in range(generations):
        new_population = []
        while len(new_population) < population_size:
            # Seleção
            parent1 = selecao_pareto(population, metricas, tournament_size)
            parent2 = selecao_pareto(population, metricas, tournament_size)
            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = [route.copy() for route in parent1], [route.copy() for route in parent2]
            # Mutação
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        # Ajusta tamanho
        population = new_population[:population_size]
        metricas = [calcula_fitness(ind, clients_coords, battery_limit) for ind in population]
        gen_best_idx = int(np.argmin([m[0] for m in metricas]))
        gen_best_fitness = metricas[gen_best_idx][0]
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_genotype = [route.copy() for route in population[gen_best_idx]]
        history.append(best_fitness)
        # Chamada do callback para animação
        if on_generation is not None:
            on_generation(gen, best_genotype, best_fitness)
    return {
        'best_genotype': best_genotype,
        'best_fitness': best_fitness,
        'history': history
    }