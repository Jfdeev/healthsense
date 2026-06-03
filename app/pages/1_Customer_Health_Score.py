"""
HealthSense — Dashboard Streamlit interativo.

Como rodar:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garante que a raiz do repo esteja no path (página em app/pages/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.fuzzy.fuzzy_engine import HealthSenseEngine
from src.fuzzy.rules import get_rules_table


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="HealthSense — Customer Health Score Fuzzy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_engine() -> HealthSenseEngine:
    return HealthSenseEngine()


engine = get_engine()


# ============================================================================
# CABEÇALHO
# ============================================================================
st.title("HealthSense — Customer Health Score (Fuzzy Mamdani)")
st.caption(
    "Sistema de apoio à decisão para Customer Success em SaaS B2B. "
    "Trabalho de Inteligência Artificial e Computacional — CESUPA 2026/1."
)

with st.expander("Sobre o produto"):
    st.markdown(
        """
**Problema:** Customer Success em SaaS B2B precisa priorizar a carteira de
clientes diariamente para agir antes do churn. Scores tradicionais são
binários (`bom/ruim`) ou somas ponderadas arbitrárias.

**Solução fuzzy:** combina 5 variáveis (engajamento, suporte, satisfação,
saúde financeira, tenure) em transições suaves usando 32 regras Mamdani
derivadas de práticas de Customer Success. A saída é um score 0-100
classificado em 5 níveis: Crítico, Em Risco, Atenção, Saudável, Promotor.
        """
    )


# ============================================================================
# SIDEBAR — ENTRADAS DO CLIENTE
# ============================================================================
st.sidebar.header("Dados do cliente")

engajamento = st.sidebar.slider("Engajamento (0–100)", 0, 100, 65, help="Índice composto: % dias ativos × adoção de features")
volume_suporte = st.sidebar.slider("Volume de Suporte (tickets/30 dias)", 0, 30, 4)
satisfacao = st.sidebar.slider("Satisfação / NPS (0–10)", 0.0, 10.0, 7.5, step=0.1)
saude_financeira = st.sidebar.slider("Atraso de pagamento (dias)", 0, 90, 0)
tenure = st.sidebar.slider("Tenure (meses de contrato)", 0, 60, 12)


# ============================================================================
# CÁLCULO PRINCIPAL
# ============================================================================
result = engine.evaluate(
    engajamento=engajamento,
    volume_suporte=volume_suporte,
    satisfacao=satisfacao,
    saude_financeira=saude_financeira,
    tenure=tenure,
)


# ============================================================================
# GAUGE PRINCIPAL + CLASSIFICAÇÃO
# ============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Score do cliente")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result.chs,
        delta={"reference": 50, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        title={"text": f"<b>{result.classificacao}</b>"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 20], "color": "#d62728"},
                {"range": [20, 40], "color": "#ff7f0e"},
                {"range": [40, 60], "color": "#f7e26b"},
                {"range": [60, 80], "color": "#2ca02c"},
                {"range": [80, 100], "color": "#1f77b4"},
            ],
        },
    ))
    gauge.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(gauge, width='stretch')

with col2:
    st.subheader("Recomendação de ação")
    acoes = {
        "Crítico": ("🚨", "Escalar imediatamente para CS sênior + retention call em 24h."),
        "Em Risco": ("⚠️", "Agendar Business Review e revisar plano de adoção esta semana."),
        "Atenção": ("👀", "Monitorar próximas 2 semanas; oferecer treinamento ou webinar."),
        "Saudável": ("✅", "Manter ritmo; abrir conversa de expansão se contexto permitir."),
        "Promotor": ("⭐", "Candidato a caso de sucesso, indicação ou upsell prioritário."),
    }
    icone, texto = acoes[result.classificacao]
    st.markdown(f"### {icone} {result.classificacao}")
    st.info(texto)

    st.metric("CHS final", f"{result.chs:.1f} / 100")


# ============================================================================
# REGRAS ATIVADAS (EXPLICABILIDADE)
# ============================================================================
st.subheader("Regras ativadas neste cenário")
ativas = sorted(
    [(rid, grau) for rid, grau in result.activations.items() if grau > 0.01],
    key=lambda x: x[1],
    reverse=True,
)
if ativas:
    df_ativas = pd.DataFrame(ativas, columns=["Regra", "Grau de ativação"])
    rule_specs = {r.rule_id: r for r in engine.rule_specs}
    df_ativas["Categoria"] = df_ativas["Regra"].map(lambda r: rule_specs[r].categoria)
    df_ativas["Justificativa"] = df_ativas["Regra"].map(lambda r: rule_specs[r].justificativa)
    st.dataframe(df_ativas, width='stretch', hide_index=True)
else:
    st.warning("Nenhuma regra foi ativada significativamente — verifique os parâmetros.")


# ============================================================================
# SUPERFÍCIE DE CONTROLE
# ============================================================================
st.subheader("Superfície de controle (análise 2D)")
st.caption("Mostra como o CHS varia ao cruzar duas variáveis, mantendo as outras fixas.")

surf_col1, surf_col2 = st.columns(2)
with surf_col1:
    x_var = st.selectbox("Eixo X", ["engajamento", "satisfacao", "saude_financeira", "volume_suporte", "tenure"], index=0)
with surf_col2:
    options_y = ["satisfacao", "engajamento", "saude_financeira", "volume_suporte", "tenure"]
    y_var = st.selectbox("Eixo Y", [v for v in options_y if v != x_var], index=0)

fixed = {
    "engajamento": engajamento,
    "volume_suporte": volume_suporte,
    "satisfacao": satisfacao,
    "saude_financeira": saude_financeira,
    "tenure": tenure,
}
fixed = {k: v for k, v in fixed.items() if k not in (x_var, y_var)}

with st.spinner("Calculando superfície..."):
    X, Y, Z = engine.control_surface(x_var, y_var, fixed, resolution=20)

surf_fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="RdYlGn", cmin=0, cmax=100)])
surf_fig.update_layout(
    scene=dict(
        xaxis_title=x_var,
        yaxis_title=y_var,
        zaxis_title="CHS",
    ),
    height=500,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(surf_fig, width='stretch')


# ============================================================================
# BASE DE REGRAS COMPLETA
# ============================================================================
with st.expander("Ver base completa de regras (32 regras)"):
    st.dataframe(pd.DataFrame(get_rules_table(engine.rule_specs)), width='stretch', hide_index=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "HealthSense • Modalidade B (Produto) • Trilha de Ampliação: Técnica • "
    "Pontuação extra: AG/PSO (ver `notebooks/04_otimizacao_pso.ipynb`)"
)
