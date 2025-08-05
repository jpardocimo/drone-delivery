"""
Módulo app.py
Interface Streamlit para o problema de entrega com drones usando Algoritmo Genético.
"""

import streamlit as st
import matplotlib.pyplot  as plt
import numpy as np
import time
import pandas as pd
from ga_core import algoritimo_genetico
from typing import List
        

DEFAULTS = {
    "num_clients":      10,
    "battery_limit":    4200.0,
    "population_size":  200,
    "generations":      200,
    "mutation_rate":    20,    #percentual %
    "crossover_rate":   80,    #percentual %
    "tournament_size":  3,
    "seed":             42,
    "elitism_percent":  10,    #percentual%
    "early_stopping":   True,
}

#coloca os valores padrão no estado da sessão
#obs: injeta apenas as chaves que ainda não existem no estado da sessão 
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# Função para gerar clientes no grafico
def gerar_coordenadas_clientes(n: int, raio: float = 800.0, seed: int = None) -> np.ndarray:
    """
    Gera coordenadas aleatórias para n clientes dentro do raio recebido como parametro.
    """
    if seed is not None:
        np.random.seed(seed)
    angulos = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n)) * raio
    x = radii * np.cos(angulos)
    y = radii * np.sin(angulos)
    return np.column_stack((x, y))

# Função para plotar as rotas dos drones
def plotar_rotas(clientes_coordenadas: np.ndarray, rotas: List[List[int]], titulo: str = "Rotas dos Drones", raio=800) -> plt.Figure:
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
    ax.scatter(0, 0, c='purple', s=150, marker='p', label='Lanchonete (0,0)')
    # Plotar clientes
    ax.scatter(clientes_coordenadas[:, 0], clientes_coordenadas[:, 1], c='black', s=60, label='Clientes')
    for i, (x, y) in enumerate(clientes_coordenadas):
        ax.text(x, y, f'{i+1}', fontsize=12, ha='right', va='bottom', color='black')
    # Cores de linhapara os drones
    colors = ['blue', 'green']
    for i, rota in enumerate(rotas):
        if not rota:
            continue
        # Construir caminho: lanchonete -> clientes -> lanchonete
        path = np.vstack(([0, 0], clientes_coordenadas[rota], [0, 0]))
        ax.plot(path[:, 0], path[:, 1], '-o', color=colors[i % len(colors)], label=f'Drone {i+1}')    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(titulo)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    ax.set_aspect('equal')
    margin = 100
    limit = raio + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    return fig

# Função para criar gráfico de convergência no estado inicial
def cria_grafico_convergencia_vazio(geracoes=200):
    """Cria gráfico de convergência vazio."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.style.use('seaborn-v0_8')
    ax.set_xlabel("Geração", fontsize=12)
    ax.set_ylabel('Distância total (m)', fontsize=12)
    ax.set_title('Convergência do AG', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, geracoes)
    ax.set_ylim(0, 5000)  
    fig.tight_layout()
    return fig

# Função para criar gráfico de convergência no estado inicial
def cria_grafico_rotas_vazio(clients_coords, title="Rotas dos Drones", raio=800):
    """Cria gráfico de rotas com apenas clientes e lanchonete (sem rotas)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    # Plotar lanchonete
    ax.scatter(0, 0, c='purple', s=150, marker='p', label='Lanchonete (0,0)')
    # Plotar clientes
    ax.scatter(clients_coords[:, 0], clients_coords[:, 1], c='black', s=60, label='Clientes')
    for i, (x, y) in enumerate(clients_coords):
        ax.text(x, y, f'{i+1}', fontsize=12, ha='right', va='bottom', color='black')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    ax.set_aspect('equal')
    margin = 100
    limit = raio + margin
    
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    return fig


# Função principal do app (Streamlit) - WEB
def main():
    """
    Função principal do app (Streamlit)
    """
    #Configuração da página
    st.title("🚁 Drone Delivery com GA - FIAP IA Devs")
    with st.expander("ℹ️ Como o Problema é Calculado", expanded=False):
        st.markdown("""
        ## 🎯 **Objetivo do Problema**
        Minimizar a **distância total** percorrida por 2 drones de entrega de uma lanchonete.
        
        ---
        
        ## 🧬 **Representação da Solução (Genoma)**
        
        **📋 Formato:** Lista com duas sublistas representando as rotas de cada drone.
        
        ```python
        # Exemplo para 6 clientes:
        [[2, 1], [3, 4, 5, 6]]
        ```
        
        **💡 Significado:**
        - 🚁 **Drone 1:** Entrega aos clientes 2 → 1 (nesta ordem)  
        - 🚁 **Drone 2:** Entrega aos clientes 3 → 4 → 5 → 6 (nesta ordem)

        ---
        
        ## 📊 **Função de Fitness**
        
        **🔢 Cálculo:** Soma das distâncias euclidianas de ambos os drones.
        
        **📍 Para cada drone:**
        Para cada drone, calcular a distância euclidiana da lanchonete ao primeiro cliente, 
        entre clientes consecutivos, e do último cliente de volta à lanchonete.
   
        
        **⚠️ Penalização:** Se distância > limite de bateria então: fitness * 100
        
        **🎯 Meta:** **Minimizar** o valor do fitness (menor distância = melhor solução)
        
        ---
        
        ## 🏆 **Método de Seleção**
        
        **🥊 Torneio (tamanho 3):**
        - Escolhe 3 indivíduos aleatoriamente (ou de acordo com o tamanho selecionado pelo usuario)
        - Seleciona o de **menor fitness**
        - Equilibra diversidade e qualidade das soluções
        
        ---
        
        ## 🔄 **Método de Crossover**
        
        **🧬 Order Crossover (OX) Adaptado:**
        - Garante permutações válidas (sem clientes repetidos)
        - Ideal para problemas de sequenciamento de rotas
        - Preserva a ordem relativa dos elementos
        
        ---
        
        ## 🎲 **Inicialização**
        
        **📝 Processo:**
        1. **Permutação aleatória** de todos os clientes
        2. **Divisão aleatória** entre os 2 drones  
                
        ```python
        # Exemplo: [[3,1,5], [2,6,4]]  #drone1: [3,1,5] e drone2: [2,6,4]
        ```
        
        ---
        
        ## 🛑 **Critérios de Parada**
        
        **📈 Opções Disponíveis:**
        - **🔢 Gerações Fixas:** Número máximo definido (ex: 200) ou 
        - **⏰ Early Stopping:** Para se sem melhoria por 30 gerações
        
        """)

    st.set_page_config(page_title="GA Drones Delivery com Algoritmo Genético - FIAP IA Devs", layout="wide", page_icon="")   

    #Inicia os valores padrão (DEFAULTS) no estado da sessão
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    #------------------------------------------------
    # Configuração da sidebar - controles da barra lateral para configurar os parametros do GA
    #------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        if st.button("🎲 Randomizar Clientes"):
            st.session_state['num_clients'] = st.session_state.get('num_clients', 12)
            st.session_state['seed'] = np.random.randint(0, 100000)
            st.rerun()
        st.markdown("---")
        resetButton = st.button("🔄 Resetar parâmetros")
        st.markdown("---")
        
        if resetButton:
            st.session_state.clear()
            st.session_state['seed'] = np.random.randint(0, 100000)
            st.rerun()
            
        st.header("Parâmetros do Problema")
        num_clients = st.slider("Número de clientes", 5, 15,  key='num_clients')
        battery_limit = st.number_input("Limite de bateria por drone (m)", 1600.0, 20000.0, step=100.0, key='battery_limit')
        
        st.header("Parâmetros do AG")
        population_size = st.slider("Tamanho da população", 10, 200, key='population_size')
        generations = st.slider("Gerações", 10, 500,step=10, key='generations')
        crossover_rate_percent = st.sidebar.slider("Probabilidade de Crossover", 0.5, 1.0, 0.8, step=0.01, help="Chance de combinar rotas(recombinação genética). Valores altos aumentam a exploração.")
        mutation_rate_percent = st.sidebar.slider("Probabilidade de Mutação", 0.0, 1.0, 0.2, step=0.01, help="Chance de alterar rotas (introduzir novidades). Valores altos aumentam a diversidade.")
        tournament_size = st.slider("Tamanho do torneio", 2, 6,  key='tournament_size')
        elitism_percent = st.selectbox(
            "Elitismo (%)", 
            options=[0, 5, 10, 15, 20], 
            index=2,  # Default: 10%
            help="Percentual dos melhores indivíduos preservados para a próxima geração"
        )
        early_stopping = st.checkbox(
            "Parada antecipada", 
            value=True,
            help="Interrompe o algoritmo se não houver melhoria por 30 gerações consecutivas"
        )
        st.markdown("---")
        botao_executar_ga_clicado = st.button("Executar Algoritmo Genético")

    #Gera os coordadas dos pontso dos clientes
    seed = st.session_state.get('seed', 42)
    num_clients_atual = st.session_state.get('num_clients', 12)
    coordenadas_clientes = gerar_coordenadas_clientes(num_clients_atual, raio=800.0, seed=int(seed))
    col_left,  col_right = st.columns([1, 1])

    #Mostra os gráficos de rotas e convergência no estado inicial
    with col_left:
        st.markdown("Rotas dos Drones")
        route_placeholder = st.empty()
        # Mostra gráfico base com clientes posicionados, porém sem as rotas ainda
        base_routes_fig = cria_grafico_rotas_vazio(coordenadas_clientes, raio=800)
        route_placeholder.pyplot(base_routes_fig, use_container_width=True)

    with col_right:
        st.markdown("Convergência do AG")
        convergence_placeholder = st.empty()
        # Mostra gráfico de convergência, porém ainda sem a linha de convergência
        empty_conv_fig = cria_grafico_convergencia_vazio(generations)
        convergence_placeholder.pyplot(empty_conv_fig, use_container_width=True)

    #Se o botão de executar for clicado, executa o algoritmo genético
    if botao_executar_ga_clicado:
        progress_bar = st.progress(0)
        historico_convergencia = []
        solutions_data = []  # Lista para armazenar dados das soluções
        
        # Tabela para visualizar soluções
        st.markdown("### 📊 Evolução das Soluções")
        table_placeholder = st.empty()
        
        #Definição da função que será chamada a cada geração (Aqui é somaente a definiçao,
        # ela é passada como callback para a funcao algoritimo_genetico e vai ser chamada de lá)
        def on_generation(gen, best_genotype, best_fitness):
            historico_convergencia.append(best_fitness)
            
            # Calcular distâncias individuais dos drones
            from ga_core import calcula_distancia_euclidiana_rota
            drone1_dist = calcula_distancia_euclidiana_rota(best_genotype[0], coordenadas_clientes) if best_genotype[0] else 0.0
            drone2_dist = calcula_distancia_euclidiana_rota(best_genotype[1], coordenadas_clientes) if best_genotype[1] else 0.0
            
            # Formatar rotas para exibição compacta
            drone1_route = ' → '.join([str(idx+1) for idx in best_genotype[0]]) if best_genotype[0] else "Vazio"
            drone2_route = ' → '.join([str(idx+1) for idx in best_genotype[1]]) if best_genotype[1] else "Vazio"
            
            # Adicionar dados da solução atual
            solution_data = {
                'Geração': gen + 1,
                'Fitness Total': f"{best_fitness:.1f}m",
                'Drone 1': drone1_route,
                'Dist. Drone1': f"{drone1_dist:.1f}m",
                'Drone 2': drone2_route, 
                'Dist. Drone2': f"{drone2_dist:.1f}m",
                'Status': '⚠️' if (drone1_dist > battery_limit or drone2_dist > battery_limit) else '✅'
            }
            solutions_data.append(solution_data)
            
            # Atualizar tabela (mostrar apenas as últimas 5 soluções, mais recente no topo)
            df = pd.DataFrame(solutions_data)  
            df = df.iloc[::-1]  # Inverter ordem - mais recente no topo
            table_placeholder.dataframe(df, use_container_width=True, hide_index=True, height=250)

            # Gráfico de rotas em tempo real
            fig_route = plotar_rotas(coordenadas_clientes, best_genotype, titulo=f"Rotas - Geração {gen+1}", raio=800)
            fig_route.set_size_inches(6, 4)
            route_placeholder.pyplot(fig_route, use_container_width=True)

            progress_bar.progress((gen +1)/ generations)
        
            fig_conv, ax_conv = plt.subplots(figsize=(6, 4))
            plt.style.use('seaborn-v0_8')
            ax_conv.plot(historico_convergencia, label='Melhor fitness')
            ax_conv.set_xlabel("Geração", fontsize=12)
            ax_conv.set_ylabel('Distância total (m)')
            ax_conv.set_title('Convergência do AG', fontsize=12)
            ax_conv.legend(fontsize=10)
            ax_conv.grid(True, linestyle='--', alpha=0.7)
            ax_conv.tick_params(labelsize=10)
            fig_conv.tight_layout()
            convergence_placeholder.pyplot(fig_conv, use_container_width=True)
            time.sleep(0.05)

        #------------------------------------------------
        # ACIONA A FUNÇÃO DO GA -  aqui é que chama o algoritmo genetico que está em ga_core.py
        #------------------------------------------------
        result = algoritimo_genetico(
            clientes_coordenadas=coordenadas_clientes,
            tamanho_populacao=population_size,
            numero_de_geracoes=generations,
            limite_bateria=battery_limit,
            taxa_de_mutacao=mutation_rate_percent,
            taxa_de_crossover=crossover_rate_percent,
            tamanho_do_torneio=tournament_size,
            percentual_de_elitismo=elitism_percent,
            parada_antecipada=early_stopping,
            callback_on_generation=on_generation  # Passa a on_generation como callback pra ser chamada de lá
        )

        melhor_individuo = result['melhor_individuo']
        melhor_fitness = result['melhor_fitness']
        stopped_early = result['stopped_early']
        geracao_final = result['geracao_final']

        # Informação sobre early stopping
        if stopped_early:
            
            # Toast notification (modal-like)
            st.toast(f"🛑 Early Stopping Ativado! Parou na geração {geracao_final}", icon="⚠️")
            
            # Alerta visual para avisar do early stopping
            st.warning(f"🛑 **Algoritmo parou antecipadamente na geração {geracao_final}**\n\n"
                      f"📊 Motivo: Sem melhoria por 30 gerações consecutivas\n\n"
                      f"💡 Isso indica que o algoritmo convergiu para uma solução estável.")
        else:
            st.success(f"✅ Algoritmo executou todas as {geracao_final} gerações com sucesso!")

        #------------------------------------------------
        # Exibição dos resultados finais
        #------------------------------------------------
        st.subheader("Melhor Plano de vôo encontrado")

        for i, route in enumerate(melhor_individuo):
            if route:
                rota_str = 'Lanchonete → ' + ' → '.join([f"Cliente {idx+1}" for idx in route]) + ' → Lanchonete'
            else:
                rota_str = 'Lanchonete → Lanchonete'
            st.write(f"**Drone {i+1}:** {rota_str}")
            dist = 0.0
            if route:
                from ga_core import calcula_distancia_euclidiana_rota
                dist = calcula_distancia_euclidiana_rota(route, coordenadas_clientes)
            st.write(f"Distância da rota: {dist:.1f} m {'⚠️' if dist > battery_limit else ''}")

        st.write(f"**Distância total percorrida:** {melhor_fitness:.1f} m")

        st.success("Execução finalizada!")
    else:
        st.info("Ajuste os parâmetros e clique em 'Executar Algoritmo Genético'.")


if __name__ == "__main__":
    main()