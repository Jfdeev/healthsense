"""
HealthSense Allocator — Dashboard Streamlit.

Produto: ferramenta de apoio à decisão que aloca a capacidade do time de
Customer Success (qual CSM atende qual cliente com qual playbook) para
maximizar a receita esperada retida/expandida, usando um Algoritmo Genético
cuja função de aptidão é o motor fuzzy de Customer Health Score (Parte 1).

Rodar:
    streamlit run src/evolutionary/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evolutionary.problem import make_default_problem, default_csms
from src.evolutionary.ga import GeneticAlgorithm, GAConfig
from src.evolutionary.memetic import MemeticAlgorithm
from src.evolutionary import baselines


st.set_page_config(page_title="HealthSense Allocator", page_icon="🧬", layout="wide")


# ============================================================================
# CACHE: o pré-cômputo fuzzy (tabela de valor) é o gargalo. Cacheamos por
# (n_customers, seed) para não recomputar a cada interação.
# ============================================================================
@st.cache_resource(show_spinner="Construindo carteira e pré-computando avaliações fuzzy...")
def get_problem(n_customers: int, seed: int):
    return make_default_problem(n_customers=n_customers, seed=seed)


def aplicar_capacidade(problem, fator: float):
    """Reescala os orçamentos dos CSMs sem recomputar a tabela fuzzy."""
    base = default_csms()
    for csm, b in zip(problem.csms, base):
        csm.orcamento_horas = b.orcamento_horas * fator
    return problem


# ============================================================================
# CABEÇALHO
# ============================================================================
st.title("🧬 HealthSense Allocator")
st.caption(
    "Otimização da alocação do time de Customer Success via Algoritmo Genético "
    "com aptidão fuzzy. Parte 2 — IA Evolutiva e Computação Bioinspirada (CESUPA 2026/1)."
)

with st.expander("Sobre o produto e o problema"):
    st.markdown(
        """
**Problema (apoio à decisão):** um time de Customer Success tem **capacidade
limitada de horas**. Para cada cliente da carteira, é preciso decidir **qual CSM
o atende com qual playbook** (intervenção) — ou nenhuma ação — de forma a
**maximizar a receita esperada retida/expandida**, respeitando o orçamento de
horas de cada CSM.

**Por que Computação Evolutiva?** É um *Problema de Atribuição Generalizada*
(GAP), NP-difícil: múltiplas restrições de capacidade + eficácia que depende do
trio (cliente, CSM, playbook) por causa das especialidades + objetivo **não-linear**
(o ganho de saúde é calculado pelo motor **fuzzy** da Parte 1). Um Algoritmo
Genético navega esse espaço combinatório melhor que heurísticas gulosas.

**Integração Fuzzy-Evolutiva:** a função de aptidão usa o sistema fuzzy Mamdani
de 32 regras (Customer Health Score) para prever o ΔCHS de cada intervenção.
        """
    )


# ============================================================================
# SIDEBAR — CONFIGURAÇÃO
# ============================================================================
st.sidebar.header("Carteira")
n_customers = st.sidebar.select_slider("Nº de clientes", options=[20, 30, 40, 60], value=40)
seed_portfolio = st.sidebar.number_input("Semente da carteira", 0, 999, 7)

st.sidebar.header("Capacidade do time")
fator_cap = st.sidebar.slider("Fator de capacidade dos CSMs", 0.5, 2.0, 1.0, 0.1,
                              help="Multiplica o orçamento de horas de todos os CSMs")

st.sidebar.header("Algoritmo")
algo = st.sidebar.selectbox("Otimizador", ["Algoritmo Genético", "Memético (AG + busca local)"])
pop = st.sidebar.slider("População", 20, 150, 80, 10)
gens = st.sidebar.slider("Gerações", 20, 250, 120, 10)
seed_algo = st.sidebar.number_input("Semente do otimizador", 0, 999, 0)

problem = get_problem(n_customers, seed_portfolio)
aplicar_capacidade(problem, fator_cap)

st.sidebar.markdown("---")
st.sidebar.metric("Capacidade total", f"{problem.capacidade_total:.0f} h")
st.sidebar.metric("Clientes na carteira", problem.N)


# ============================================================================
# EXECUÇÃO
# ============================================================================
col_run, col_info = st.columns([1, 3])
rodar = col_run.button("▶ Otimizar alocação", type="primary", width='stretch')

if rodar:
    cfg = GAConfig(seed=seed_algo, pop_size=pop, n_generations=gens)
    with st.spinner(f"Rodando {algo}..."):
        if algo.startswith("Memético"):
            result = MemeticAlgorithm(problem, cfg).run()
        else:
            result = GeneticAlgorithm(problem, cfg).run()
        greedy = baselines.greedy(problem)
        rand = baselines.random_search(problem, n_samples=pop * gens, seed=seed_algo)

    st.session_state["result"] = result
    st.session_state["greedy"] = greedy
    st.session_state["rand"] = rand


# ============================================================================
# RESULTADOS
# ============================================================================
if "result" in st.session_state:
    result = st.session_state["result"]
    greedy = st.session_state["greedy"]
    rand = st.session_state["rand"]
    res = result.best_result

    st.subheader("Resultado da otimização")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Receita esperada (R$)", f"{res.valor_total:,.0f}",
              delta=f"{(result.best_fitness/greedy.best_fitness-1)*100:+.1f}% vs guloso")
    c2.metric("Clientes atendidos", f"{res.n_atendidos} / {problem.N}")
    c3.metric("Avaliações da aptidão", f"{result.evaluations:,}")
    c4.metric("Tempo", f"{result.wall_time:.1f}s")

    # --- Comparação com baselines ---
    st.markdown("#### Comparação com métodos simples")
    comp = pd.DataFrame([
        {"Método": "Sem otimização", "Receita (R$)": 0, "Atendidos": 0},
        {"Método": "Busca aleatória", "Receita (R$)": rand.best_result.valor_total, "Atendidos": rand.best_result.n_atendidos},
        {"Método": "Guloso (densidade)", "Receita (R$)": greedy.best_result.valor_total, "Atendidos": greedy.best_result.n_atendidos},
        {"Método": result.algoritmo, "Receita (R$)": res.valor_total, "Atendidos": res.n_atendidos},
    ])
    bar = go.Figure(go.Bar(
        x=comp["Método"], y=comp["Receita (R$)"],
        marker_color=["#bbb", "#ff7f0e", "#1f77b4", "#2ca02c"],
        text=[f"R$ {v:,.0f}" for v in comp["Receita (R$)"]], textposition="outside",
    ))
    bar.update_layout(height=320, yaxis_title="Receita esperada (R$)", margin=dict(t=20, b=20))
    st.plotly_chart(bar, width='stretch')

    # --- Curva de convergência ---
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Curva de convergência")
        conv = go.Figure()
        conv.add_trace(go.Scatter(y=result.history_best, mode="lines", name="Melhor", line=dict(color="#2ca02c")))
        conv.add_trace(go.Scatter(y=result.history_mean, mode="lines", name="Média da população", line=dict(color="#aaa", dash="dot")))
        conv.add_hline(y=greedy.best_fitness, line_dash="dash", line_color="#1f77b4",
                       annotation_text="Guloso", annotation_position="bottom right")
        conv.update_layout(height=320, xaxis_title="Geração", yaxis_title="Aptidão (R$)", margin=dict(t=20))
        st.plotly_chart(conv, width='stretch')

    with colB:
        st.markdown("#### Horas alocadas por CSM")
        horas = res.horas_por_csm
        orcamentos = {c.nome: c.orcamento_horas for c in problem.csms}
        hb = go.Figure()
        hb.add_trace(go.Bar(x=list(horas.keys()), y=list(horas.values()), name="Usado", marker_color="#2ca02c"))
        hb.add_trace(go.Bar(x=list(orcamentos.keys()), y=list(orcamentos.values()), name="Orçamento", marker_color="rgba(0,0,0,0.15)"))
        hb.update_layout(barmode="overlay", height=320, yaxis_title="Horas", margin=dict(t=20))
        st.plotly_chart(hb, width='stretch')

    # --- Plano de ação (alocação detalhada) ---
    st.markdown("#### Plano de ação recomendado")
    df = pd.DataFrame(res.detalhe)
    df_acao = df[df["acao"] != "Nenhuma ação"].sort_values("valor", ascending=False)
    st.dataframe(
        df_acao.rename(columns={
            "cliente": "Cliente", "mrr": "MRR", "chs_atual": "CHS atual",
            "classe_atual": "Classe", "acao": "Ação (CSM → Playbook)",
            "delta_chs": "ΔCHS", "horas": "Horas", "valor": "Valor (R$)",
        }),
        width='stretch', hide_index=True,
    )

    # --- Exportação (módulo de produto) ---
    csv = df_acao.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Exportar plano de ação (CSV)", csv, "plano_acao.csv", "text/csv")

    nao_atendidos = df[df["acao"] == "Nenhuma ação"]
    if len(nao_atendidos):
        with st.expander(f"Clientes sem ação ({len(nao_atendidos)}) — capacidade insuficiente"):
            st.dataframe(nao_atendidos[["cliente", "mrr", "chs_atual", "classe_atual"]],
                         width='stretch', hide_index=True)
else:
    st.info("Configure os parâmetros na barra lateral e clique em **Otimizar alocação**.")

st.markdown("---")
st.caption("HealthSense Allocator • AG com aptidão fuzzy • Ampliação 5p: comparação AG vs Memético • Extra: integração Fuzzy-Evolutiva")
