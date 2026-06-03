"""
HealthSense — plataforma unificada de Customer Success para SaaS B2B.

Produto único com dois módulos:
  1. Customer Health Score (Parte 1) — sistema fuzzy Mamdani que pontua a saúde
     de cada cliente.
  2. Allocator (Parte 2) — otimizador evolutivo que decide o plano de ação do
     time de CS, usando o score fuzzy como função de aptidão.

Rodar:
    streamlit run app/Home.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="HealthSense", page_icon="🩺", layout="wide")

st.title("🩺 HealthSense")
st.subheader("Plataforma de Customer Success para SaaS B2B")
st.caption(
    "Inteligência Artificial e Computacional (0700M8) — CESUPA 2026/1 · "
    "Prof. Daniel Leal Souza"
)

st.markdown(
    """
**Um produto, duas capacidades.** O HealthSense ajuda times de Customer Success
a **enxergar** a saúde da carteira e a **agir** sobre ela com recursos limitados.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Módulo 1 — Customer Health Score")
    st.markdown(
        """
*Parte 1 — Sistemas de Controle Fuzzy (Mamdani)*

Pontua a saúde de cada cliente (0–100) a partir de 5 sinais —
engajamento, suporte, satisfação, financeiro e tenure — usando uma base de
**32 regras fuzzy** com transições suaves e explicabilidade.

**Responde:** *quão saudável está cada cliente?*
        """
    )
    st.page_link("pages/1_Customer_Health_Score.py", label="Abrir o Scorer", icon="📊")

with col2:
    st.markdown("### 🧬 Módulo 2 — Allocator")
    st.markdown(
        """
*Parte 2 — IA Evolutiva e Computação Bioinspirada*

Decide **qual CSM atende qual cliente com qual playbook** para maximizar a
receita esperada retida/expandida, via **Algoritmo Genético** cuja função de
aptidão é o motor fuzzy do Módulo 1.

**Responde:** *com horas limitadas, qual o melhor plano de ação?*
        """
    )
    st.page_link("pages/2_Allocator.py", label="Abrir o Allocator", icon="🧬")

st.divider()

with st.expander("Como os módulos se conectam"):
    st.markdown(
        """
O Allocator pergunta, para cada intervenção candidata: *"se eu fizer esta ação,
quanto a saúde do cliente melhora?"*. Quem responde é o **motor fuzzy** do
Módulo 1 (`src/fuzzy/`). Assim, o ganho de saúde (ΔCHS) previsto pelo sistema
fuzzy vira o **valor** que o Algoritmo Genético otimiza.

```
Allocator (AG)  →  "e se eu fizer esta ação?"  →  Fuzzy CHS (32 regras)  →  ΔCHS  →  valor
```

Essa integração Fuzzy–Evolutiva é também a base da pontuação extra da Parte 2.
        """
    )

st.caption(
    "Navegue pelos módulos na barra lateral. Código: `src/fuzzy/` (Parte 1) e "
    "`src/evolutionary/` (Parte 2)."
)
