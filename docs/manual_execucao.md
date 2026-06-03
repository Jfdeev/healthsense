# Manual de Execução — HealthSense (unificado)

## 1. Requisitos
- Python 3.10+
- pip
- Navegador (para o app)

## 2. Instalação

```bash
git clone <url-do-repo>
cd healthsense

python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Produto (app unificado)

```bash
streamlit run app/Home.py
```

Abre em `http://localhost:8501`. Barra lateral:
- **Home** — visão geral e como os módulos se conectam.
- **Customer Health Score** (Parte 1) — mova os sliders das 5 entradas; veja o gauge do CHS, a recomendação, as regras ativadas e a superfície de controle.
- **Allocator** (Parte 2) — configure carteira e capacidade, escolha o otimizador (AG/Memético) e clique **Otimizar alocação**; veja receita vs. baselines, convergência, horas por CSM e exporte o plano (CSV).

> A primeira execução do Allocator pré-computa as avaliações fuzzy (alguns segundos para 40 clientes).

## 4. Testes (32)

```bash
pytest tests/ -v
# tests/test_fuzzy.py ......... 16 (Parte 1)
# tests/test_evolutionary.py .. 16 (Parte 2)
```

## 5. Reproduzir figuras e tabelas

```bash
python experiments/parte1_figuras.py        # Parte 1: MFs, superfícies, sensibilidade
python experiments/parte1_pso.py            # Parte 1 (extra): PSO das MFs
python experiments/parte2_experimentos.py   # Parte 2: AG vs baselines, AG vs memético, capacidade, fuzzy vs linear
```

Saídas: `assets/parte1/`, `assets/parte2/`, `data/`.

## 6. Uso via código (API)

```python
# Parte 1 — pontuar um cliente
from src.fuzzy.fuzzy_engine import HealthSenseEngine
eng = HealthSenseEngine()
print(eng.evaluate(engajamento=80, volume_suporte=2, satisfacao=9,
                   saude_financeira=0, tenure=18))

# Parte 2 — otimizar a alocação
from src.evolutionary.problem import make_default_problem
from src.evolutionary.ga import run_ga
prob = make_default_problem(n_customers=40, seed=7)
ga = run_ga(prob, seed=0, pop_size=80, n_generations=150)
print(f"Receita: R$ {ga.best_result.valor_total:,.0f} | atendidos {ga.best_result.n_atendidos}/{prob.N}")
```

## 7. Problemas comuns

| Sintoma | Solução |
|---|---|
| `ModuleNotFoundError: skfuzzy` | Ative o venv e `pip install -r requirements.txt` |
| App não abre o navegador | Acesse `http://localhost:8501` manualmente |
| PowerShell bloqueia o venv | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError: src` ao rodar scripts | Rode a partir da raiz do repo (`healthsense/`) |
