"""
Módulo app.py
Interface Streamlit para o problema de entrega com drones usando Algoritmo Genético.
"""

import streamlit as st
import matplotlib.pyplot  as plt
import numpy as np
from ga_core import genetic_algorithm
from typing import List


def gerar_clientes(n: int, raio: float = 500.0, seed: int = None) -> np.ndarray:
    """
    Gera coordenadas aleatórias para n clientes dentro de um círculo de raio dado.
    """
    if seed is not None:
        np.random.seed(seed)
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n)) * raio
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))


def plot_routes(clients_coords: np.ndarray, routes: List[List[int]], title: str = "Rotas dos Drones") -> plt.Figure:
    """
    Plota as rotas dos drones em um gráfico 2D.
    Args:
        clients_coords: array de coordenadas dos clientes (shape: [n, 2])
        routes: lista de duas sublistas, cada uma com a ordem dos clientes visitados por cada drone
        title: título do gráfico
    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    # Plotar lanchonete
    ax.scatter(0, 0, c='red', s=120, marker='*', label='Lanchonete (0,0)')
    # Plotar clientes
    ax.scatter(clients_coords[:, 0], clients_coords[:, 1], c='black', s=60, label='Clientes')
    for i, (x, y) in enumerate(clients_coords):
        ax.text(x, y, f'{i+1}', fontsize=12, ha='right', va='bottom', color='black')
    # Cores para os drones
    colors = ['blue', 'green']
    for i, route in enumerate(routes):
        if not route:
            continue
        # Construir caminho: lanchonete -> clientes -> lanchonete
        path = np.vstack(([0, 0], clients_coords[route], [0, 0]))
        ax.plot(path[:, 0], path[:, 1], '-o', color=colors[i % len(colors)], label=f'Drone {i+1}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    ax.set_aspect('equal')
    return fig

def main():
    """
    Função principal do app Streamlit.
    """
    st.title("🚁 Drone Delivery com Algoritmo Genético")
    with st.expander("ℹ️ Como o Problema é Calculado", expanded=False):
        st.markdown("""
        **Problema:**
        - 2 drones partem da lanchonete (0,0), entregam lanches a clientes e retornam.
        - Cada drone tem limite de bateria (distância máxima).
        - Objetivo: **minimizar a distância total percorrida**.
        """)

    st.set_page_config(page_title="GA Drones Delivery com Algoritmo Genético - FIAP IA Devs", layout="wide", page_icon="")   

    if 'num_clients' not in st.session_state:
        st.session_state['num_clients'] = 12
    if 'battery_limit' not in st.session_state:
        st.session_state['battery_limit'] = 3000.0
    if 'population_size' not in st.session_state:
        st.session_state['population_size'] = 100
    if 'generations' not in st.session_state:
        st.session_state['generations'] = 200
    if 'mutation_rate' not in st.session_state:
        st.session_state['mutation_rate'] = 20
    if 'crossover_rate' not in st.session_state:
        st.session_state['crossover_rate'] = 80
    if 'tournament_size' not in st.session_state:
        st.session_state['tournament_size'] = 3

    with st.sidebar:
        st.markdown("---")
        resetButton = st.button("🔄 Resetar parâmetros")
        st.markdown("---")
        
        if resetButton:
            st.session_state.clear()
            st.session_state['seed'] = np.random.randint(0, 100000)
            st.rerun()
            
        st.header("Parâmetros do Problema")
        num_clients = st.slider("Número de clientes", 5, 15,  key='num_clients')
        battery_limit = st.number_input("Limite de bateria por drone (m)", 1600.0, 5000.0, step=100.0, key='battery_limit')
        seed = 42 #st.number_input("Seed aleatória (opcional)", value=42,step=1)
        
        st.header("Parâmetros do AG")
        population_size = st.slider("Tamanho da população", 10, 200, key='population_size')
        generations = st.slider("Gerações", 10, 500,step=10, key='generations')
        crossover_rate_percent = st.sidebar.slider("Probabilidade de Crossover", 0.5, 1.0, 0.7, step=0.05, help="Chance de combinar rotas(recombinação genética). Valores altos aumentam a exploração.")
        mutation_rate_percent = st.sidebar.slider("Probabilidade de Mutação", 0.5, 1.0, 0.9, step=0.05, help="Chance de alterar rotas (introduzir novidades). Valores altos aumentam a diversidade.")
        tournament_size = st.slider("Tamanho do torneio", 2, 6,  key='tournament_size')
        st.markdown("---")
        run = st.button("Executar Algoritmo Genético")

    


    if run:
        clients_coords = gerar_clientes(num_clients, raio=800.0, seed=int(seed))
        col_left,  col_right = st.columns([1, 1])
        progress_bar = st.progress(0)
        import time
        convergence_history = []

        with col_left:
            route_placeholder = st.empty()  # Espaço reservado para o gráfico animado de rotas
        with col_right:
            convergence_placeholder = st.empty()  # Espaço reservado para o gráfico de convergência

        def on_generation(gen, best_genotype, best_fitness):
            convergence_history.append(best_fitness)
            # Gráfico de rotas
            fig_route = plot_routes(clients_coords, best_genotype, title=f"Rotas - Geração {gen+1}")
            fig_route.set_size_inches(6, 4)
            route_placeholder.pyplot(fig_route, use_container_width=True)

            progress_bar.progress((gen +1)/ generations)
        
            fig_conv, ax_conv = plt.subplots(figsize=(6, 4))
            plt.style.use('seaborn-v0_8')
            ax_conv.plot(convergence_history, label='Melhor fitness')
            ax_conv.set_xlabel("Geração", fontsize=12)
            ax_conv.set_ylabel('Distância total (m)')
            ax_conv.set_title('Convergência do AG', fontsize=12)
            ax_conv.legend(fontsize=10)
            ax_conv.grid(True, linestyle='--', alpha=0.7)
            ax_conv.tick_params(labelsize=10)
            fig_conv.tight_layout()
            convergence_placeholder.pyplot(fig_conv, use_container_width=True)
            time.sleep(0.05)

        # crossover_rate = crossover_rate_percent / 100
        # mutation_rate = mutation_rate_percent / 100

        result = genetic_algorithm(
            clients_coords=clients_coords,
            population_size=population_size,
            generations=generations,
            battery_limit=battery_limit,
            mutation_rate=mutation_rate_percent,
            crossover_rate=crossover_rate_percent,
            tournament_size=tournament_size,
            on_generation=on_generation  # Passa o callback
        )
        best_routes = result['best_genotype']
        best_fitness = result['best_fitness']
        history = result['history']

        st.subheader("Melhor Solução Encontrada")
        for i, route in enumerate(best_routes):
            rota_str = ' → '.join([f"Cliente {idx+1}" for idx in route])
            st.write(f"**Drone {i+1}:** {rota_str if rota_str else 'Nenhum cliente'}")
            dist = 0.0
            if route:
                from ga_core import route_distance
                dist = route_distance(route, clients_coords)
            st.write(f"Distância da rota: {dist:.1f} m {'⚠️' if dist > battery_limit else ''}")
        st.write(f"**Distância total percorrida:** {best_fitness:.1f} m")

        # st.subheader("Resultados Finais")

        # col_final1, col_final2 = st.columns(2)

        # with col_final1:
        #     st.markdown("**Rotas dos Drones (Final)**")
        #     fig = plot_routes(clients_coords, best_routes)
        #     fig.set_size_inches(5, 5)
        #     st.pyplot(fig, use_container_width=True)

        # with col_final2:
        #     st.markdown("**Convergência do Algoritmo (Final)**")
        #     import pltas plt
        #     fig2, ax2 = plt.subplots(figsize=(5, 5))
        #     ax2.plot(history, label='Melhor fitness')
        #     ax2.set_xlabel('Geração')
        #     ax2.set_ylabel('Distância total (m)')
        #     ax2.set_title('Convergência do AG')
        #     ax2.grid(True)
        #     st.pyplot(fig2, use_container_width=True)

        st.success("Execução finalizada!")
    else:
        st.info("Ajuste os parâmetros e clique em 'Executar Algoritmo Genético'.")


if __name__ == "__main__":
    main()