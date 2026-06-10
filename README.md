# 🩺 HealthSense — Plataforma de Customer Success para SaaS B2B

> **Um produto, duas capacidades.** O HealthSense ajuda times de Customer Success a **enxergar** a saúde da carteira (score fuzzy) e a **agir** sobre ela com recursos limitados (otimização evolutiva).

**Disciplina:** Inteligência Artificial e Computacional (0700M8) — CESUPA · Argo
**Professor:** Daniel Leal Souza · **Semestre:** 2026/1

Este repositório reúne, como **um único produto**, os dois trabalhos da disciplina:

| Módulo | Trabalho | Pergunta que responde | Código |
|---|---|---|---|
| 📊 **Customer Health Score** | **Parte 1** — Sistemas de Controle Fuzzy | *Quão saudável está cada cliente?* | [`src/fuzzy/`](src/fuzzy/) |
| 🧬 **Allocator** | **Parte 2** — IA Evolutiva e Computação Bioinspirada | *Com horas limitadas, qual o melhor plano de ação?* | [`src/evolutionary/`](src/evolutionary/) |

O Allocator usa o motor fuzzy do Scorer como **função de aptidão** — é a integração Fuzzy–Evolutiva que rende a pontuação extra da Parte 2.

> **Nota de avaliação:** embora seja um produto único, as duas partes são entregues e avaliadas separadamente (datas e rubricas distintas). Este repositório está **organizado por parte** (veja `docs/`, `tests/`, `experiments/`, `assets/`) para que cada trabalho seja avaliável de forma independente. Cada PDF aponta para o seu módulo.

---

## Como executar

```bash
git clone <url-do-repo>
cd healthsense
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### Produto (app unificado)

```bash
streamlit run app/Home.py
```

Abre em `http://localhost:8501`. Na barra lateral:
- **Customer Health Score** — pontua um cliente (sliders → gauge + regras ativadas + superfícies).
- **Allocator** — otimiza o plano de ação do time de CS (configura carteira/capacidade → roda o AG → plano exportável).

### Testes (32 no total)

```bash
pytest tests/ -v          # tests/test_fuzzy.py (16) + tests/test_evolutionary.py (16)
```

### Reproduzir figuras e tabelas

```bash
python experiments/parte1_figuras.py        # MFs, superfícies, sensibilidade (Parte 1)
python experiments/parte1_pso.py            # extra: PSO das MFs (Parte 1)
python experiments/parte2_experimentos.py   # AG vs baselines, AG vs memético, etc. (Parte 2)
```

Saídas em `assets/parte1/`, `assets/parte2/` e `data/`.

---

## Parte 1 — Customer Health Score (Fuzzy Mamdani)

Sistema fuzzy que combina **5 entradas** (engajamento, volume de suporte, satisfação/NPS, saúde financeira, tenure) em um **Customer Health Score** de 0–100, classificado em 5 níveis.

- **32 regras** Mamdani em 7 categorias, com justificativa de domínio
- Funções de pertinência trapezoidais/triangulares/gaussianas
- Inferência min/max, defuzzificação por **centróide**
- Ampliação (equipe de 5): análise de sensibilidade · Extra: **PSO** otimiza as MFs
- Validação: 14 cenários + 16 testes

Detalhes: [`docs/Relatorio_Parte1.pdf`](docs/Relatorio_Parte1.pdf) · defesa: [`docs/roteiro_defesa_parte1.md`](docs/roteiro_defesa_parte1.md)

---

## Parte 2 — Allocator (Algoritmo Genético + aptidão fuzzy)

Otimizador que decide **qual CSM atende qual cliente com qual playbook** para maximizar a receita esperada retida/expandida, respeitando o orçamento de horas de cada CSM.

- Formulado como **Problema de Atribuição Generalizada (GAP)** — NP-difícil
- **Algoritmo Genético** (torneio, crossover uniforme, mutação, elitismo, operador de reparo)
- Aptidão = **motor fuzzy da Parte 1** (ΔCHS previsto)
- Ampliação (equipe de 5): comparação **AG vs Memético** (simples vs híbrido)
- Extra: integração **Fuzzy–Evolutiva** (fuzzy vs linear)

### Resultados (5 sementes)

| Método | Receita média (R$) |
|---|---|
| Sem otimização | 0 |
| Busca aleatória | 6.255 |
| Guloso | 9.517 |
| **Algoritmo Genético** | **9.718** (+2,1% vs guloso · +55% vs aleatório) |

**AG vs Memético:** no eixo de avaliações (o justo), o AG puro domina — o memético gasta ~4× avaliações sem melhorar. **Fuzzy vs Linear:** trocar o fuzzy por um modelo linear muda 29/40 alocações e degrada o valor real (9.747 → 3.484).

Detalhes: [`docs/Relatorio_Parte2.pdf`](docs/Relatorio_Parte2.pdf) · defesa: [`docs/roteiro_defesa_parte2.md`](docs/roteiro_defesa_parte2.md)

---

## Estrutura do repositório

```
healthsense/
├── app/                         # PRODUTO — Streamlit multipágina
│   ├── Home.py                  # visão geral + navegação
│   └── pages/
│       ├── 1_Customer_Health_Score.py   # Parte 1 (Scorer fuzzy)
│       └── 2_Allocator.py               # Parte 2 (Allocator evolutivo)
├── src/
│   ├── fuzzy/                   # Parte 1 — motor fuzzy (32 regras)
│   └── evolutionary/            # Parte 2 — GA, memético, baselines, problema GAP
├── experiments/
│   ├── parte1_figuras.py · parte1_pso.py
│   └── parte2_experimentos.py
├── tests/
│   ├── test_fuzzy.py           # 16 testes (Parte 1)
│   └── test_evolutionary.py    # 16 testes (Parte 2)
├── assets/{parte1,parte2}/     # figuras geradas
├── data/                       # tabelas (CSV) + carteira
└── docs/                       # relatórios, manuais, declarações de IA, roteiros de defesa
```

---

## Tecnologias

Python 3.10+ · scikit-fuzzy · NumPy · Pandas · Matplotlib · Plotly · Streamlit · PySwarms · pytest

## Declaração de uso de IA

Ver [`docs/declaracao_uso_ia.md`](docs/declaracao_uso_ia.md). Todo o material foi revisado, executado e validado pela equipe.
