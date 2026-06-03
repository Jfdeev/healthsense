"""
Experimentos de validação do HealthSense Allocator.

Gera TODAS as figuras (assets/) e tabelas (data/) usadas no relatório e nos
slides. Cada experimento é uma função, reproduzível por semente.

Rodar tudo:
    python experiments/run_experiments.py

Rodar um só:
    python experiments/run_experiments.py exp1

Experimentos:
    exp1  Validação: AG vs guloso vs aleatório vs sem-otimização (5 sementes)
    exp2  Ampliação obrigatória: AG vs Memético (eixo gerações E avaliações)
    exp3  Sweep de capacidade: vantagem do AG vs guloso por regime
    exp4  Extra Fuzzy-Evolutiva: impacto fuzzy vs linear na alocação
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evolutionary.problem import (
    make_default_problem, AllocationProblem,
    default_csms, default_playbooks, default_specialty_mult,
)
from src.evolutionary.ga import GeneticAlgorithm, GAConfig
from src.evolutionary.memetic import MemeticAlgorithm
from src.evolutionary import baselines
from src.fuzzy import HealthSenseEngine

ASSETS = ROOT / "assets" / "parte2"
DATA = ROOT / "data"
ASSETS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N = 40
PORTFOLIO_SEED = 7

# Motor fuzzy compartilhado (evita reconstruir; o pré-cômputo é o gargalo)
_ENGINE = HealthSenseEngine()


def _fmt(v):
    return f"{v:,.0f}"


# ============================================================================
# EXP 1 — VALIDAÇÃO: AG vs BASELINES
# ============================================================================
def exp1():
    print("[exp1] Validação AG vs baselines (5 sementes)...")
    prob = make_default_problem(n_customers=N, seed=PORTFOLIO_SEED, engine=_ENGINE)

    no = baselines.no_action(prob)
    greedy = baselines.greedy(prob)
    # Mesma configuração do AG em todos os experimentos (pop 80, 150 gerações) para
    # coerência numérica entre as tabelas do relatório.
    rand_runs = [baselines.random_search(prob, n_samples=80 * 150, seed=s) for s in SEEDS]
    ga_runs = [GeneticAlgorithm(prob, GAConfig(seed=s, pop_size=80, n_generations=150)).run() for s in SEEDS]

    def stats(vals):
        a = np.array(vals)
        return a.max(), a.mean(), a.std()

    rows = []
    rows.append(("Sem otimização", 0, 0, 0, no.best_result.n_atendidos, True))
    mx, mn, sd = stats([r.best_fitness for r in rand_runs])
    rows.append(("Busca aleatória", mx, mn, sd, int(np.mean([r.best_result.n_atendidos for r in rand_runs])), True))
    rows.append(("Guloso (densidade)", greedy.best_fitness, greedy.best_fitness, 0,
                 greedy.best_result.n_atendidos, greedy.best_result.factivel))
    mx, mn, sd = stats([r.best_fitness for r in ga_runs])
    rows.append(("Algoritmo Genético", mx, mn, sd, int(np.mean([r.best_result.n_atendidos for r in ga_runs])), True))

    df = pd.DataFrame(rows, columns=["Método", "Melhor", "Média", "Desvio", "Atendidos(méd)", "Factível"])
    df.to_csv(DATA / "exp1_validacao.csv", index=False)
    print(df.to_string(index=False))
    print(f"  AG vs guloso (média): {(df.iloc[3]['Média']/greedy.best_fitness - 1)*100:+.1f}%")

    # Figura: barras de receita média + convergência média do AG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    cores = ["#bbbbbb", "#ff7f0e", "#1f77b4", "#2ca02c"]
    ax1.bar(df["Método"], df["Média"], color=cores)
    ax1.set_ylabel("Receita esperada média (R$)")
    ax1.set_title("AG vs métodos simples (5 sementes)")
    ax1.tick_params(axis="x", rotation=20)
    for i, v in enumerate(df["Média"]):
        ax1.text(i, v, f"  {v:,.0f}", ha="center", va="bottom", fontsize=9)

    L = min(len(r.history_best) for r in ga_runs)
    H = np.array([r.history_best[:L] for r in ga_runs])
    ax2.plot(range(1, L + 1), H.mean(0), color="#2ca02c", label="AG (melhor, média de 5)")
    ax2.fill_between(range(1, L + 1), H.mean(0) - H.std(0), H.mean(0) + H.std(0), color="#2ca02c", alpha=0.2)
    ax2.axhline(greedy.best_fitness, ls="--", color="#1f77b4", label="Guloso")
    ax2.axhline(np.mean([r.best_fitness for r in rand_runs]), ls="--", color="#ff7f0e", label="Busca aleatória")
    ax2.set_xlabel("Geração")
    ax2.set_ylabel("Aptidão (R$)")
    ax2.set_title("Convergência do AG")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "exp1_validacao.png", dpi=150)
    plt.close(fig)
    print(f"  -> {ASSETS / 'exp1_validacao.png'}")


# ============================================================================
# EXP 2 — AG vs MEMÉTICO (AMPLIAÇÃO OBRIGATÓRIA)
# ============================================================================
def exp2():
    print("[exp2] AG vs Memético — eixo gerações E avaliações (5 sementes)...")
    prob = make_default_problem(n_customers=N, seed=PORTFOLIO_SEED, engine=_ENGINE)
    greedy = baselines.greedy(prob)

    ga_runs = [GeneticAlgorithm(prob, GAConfig(seed=s, pop_size=80, n_generations=150)).run() for s in SEEDS]
    mem_runs = [MemeticAlgorithm(prob, GAConfig(seed=s, pop_size=80, n_generations=150, ls_elites=3)).run() for s in SEEDS]

    def agg(runs):
        return (np.mean([r.best_fitness for r in runs]), np.std([r.best_fitness for r in runs]),
                np.mean([r.evaluations for r in runs]), np.mean([r.wall_time for r in runs]))

    rows = []
    for nome, runs in [("Algoritmo Genético", ga_runs), ("Memético (AG+busca local)", mem_runs)]:
        fm, fs, em, tm = agg(runs)
        rows.append((nome, fm, fs, em, tm))
    df = pd.DataFrame(rows, columns=["Método", "Fitness(méd)", "Desvio", "Avaliações(méd)", "Tempo(s)"])
    df.to_csv(DATA / "exp2_ag_vs_memetico.csv", index=False)
    print(df.to_string(index=False))
    print(f"  Memético custa {df.iloc[1]['Avaliações(méd)']/df.iloc[0]['Avaliações(méd)']:.1f}x avaliações "
          f"para fitness {(df.iloc[1]['Fitness(méd)']/df.iloc[0]['Fitness(méd)']-1)*100:+.1f}% vs AG")

    # Figura: convergência em DOIS eixos — gerações (enganoso) vs avaliações (honesto)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # eixo gerações
    for runs, cor, nome in [(ga_runs, "#2ca02c", "AG"), (mem_runs, "#d62728", "Memético")]:
        L = min(len(r.history_best) for r in runs)
        H = np.array([r.history_best[:L] for r in runs])
        ax1.plot(range(1, L + 1), H.mean(0), color=cor, label=nome)
    ax1.axhline(greedy.best_fitness, ls="--", color="#1f77b4", label="Guloso")
    ax1.set_xlabel("Geração"); ax1.set_ylabel("Aptidão (R$)")
    ax1.set_title("Eixo GERAÇÕES (parece que memético converge antes)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # eixo avaliações (honesto)
    for runs, cor, nome in [(ga_runs, "#2ca02c", "AG"), (mem_runs, "#d62728", "Memético")]:
        # média sobre a grade comum de avaliações
        max_ev = min(r.history_evals[-1] for r in runs)
        grid = np.linspace(0, max_ev, 100)
        curvas = []
        for r in runs:
            curvas.append(np.interp(grid, r.history_evals, r.history_best))
        M = np.array(curvas)
        ax2.plot(grid, M.mean(0), color=cor, label=nome)
    ax2.axhline(greedy.best_fitness, ls="--", color="#1f77b4", label="Guloso")
    ax2.set_xlabel("Avaliações da função objetivo"); ax2.set_ylabel("Aptidão (R$)")
    ax2.set_title("Eixo AVALIAÇÕES (honesto: AG domina)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "exp2_ag_vs_memetico.png", dpi=150)
    plt.close(fig)
    print(f"  -> {ASSETS / 'exp2_ag_vs_memetico.png'}")


# ============================================================================
# EXP 3 — SWEEP DE CAPACIDADE
# ============================================================================
def exp3():
    print("[exp3] Sweep de capacidade (vantagem AG vs guloso por regime)...")
    prob = make_default_problem(n_customers=N, seed=PORTFOLIO_SEED, engine=_ENGINE)
    custs = prob.customers

    fatores = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    rows = []
    for fator in fatores:
        csms = default_csms()
        for c in csms:
            c.orcamento_horas *= fator
        p = AllocationProblem(custs, csms, default_playbooks(), default_specialty_mult(), engine=_ENGINE)
        g = baselines.greedy(p).best_fitness
        ga = np.mean([GeneticAlgorithm(p, GAConfig(seed=s, pop_size=80, n_generations=150)).run().best_fitness
                      for s in [0, 1, 2]])
        rows.append((fator, p.capacidade_total, g, ga, (ga / g - 1) * 100))
        print(f"  fator={fator:.1f} cap={p.capacidade_total:.0f}h guloso={g:,.0f} AG={ga:,.0f} ({(ga/g-1)*100:+.1f}%)")

    df = pd.DataFrame(rows, columns=["Fator", "Capacidade(h)", "Guloso", "AG", "AG vs guloso(%)"])
    df.to_csv(DATA / "exp3_capacidade.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["Capacidade(h)"], df["Guloso"], "o-", color="#1f77b4", label="Guloso")
    ax.plot(df["Capacidade(h)"], df["AG"], "s-", color="#2ca02c", label="AG")
    ax.set_xlabel("Capacidade total do time (horas)")
    ax.set_ylabel("Receita esperada (R$)")
    ax.set_title("AG vs Guloso por regime de capacidade")
    ax.legend(); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(df["Capacidade(h)"], df["AG vs guloso(%)"], width=6, alpha=0.15, color="#2ca02c")
    ax2.set_ylabel("Vantagem do AG (%)")
    fig.tight_layout()
    fig.savefig(ASSETS / "exp3_capacidade.png", dpi=150)
    plt.close(fig)
    print(f"  -> {ASSETS / 'exp3_capacidade.png'}")


# ============================================================================
# EXP 4 — FUZZY vs LINEAR (EXTRA)
# ============================================================================
def exp4():
    print("[exp4] Impacto Fuzzy vs Linear (integração Fuzzy-Evolutiva)...")
    prob_f = make_default_problem(n_customers=N, seed=PORTFOLIO_SEED, fitness_mode="fuzzy", engine=_ENGINE)
    prob_l = make_default_problem(n_customers=N, seed=PORTFOLIO_SEED, fitness_mode="linear", engine=_ENGINE)

    ga_f = GeneticAlgorithm(prob_f, GAConfig(seed=0, pop_size=80, n_generations=150)).run()
    ga_l = GeneticAlgorithm(prob_l, GAConfig(seed=0, pop_size=80, n_generations=150)).run()

    # Avalia AMBAS alocações sob o objetivo VERDADEIRO (fuzzy)
    val_f = prob_f.evaluate(ga_f.best_genome).valor_total
    val_l = prob_f.evaluate(ga_l.best_genome).valor_total
    difere = int(np.sum(ga_f.best_genome != ga_l.best_genome))

    print(f"  Alocação guiada por FUZZY  -> valor real: {val_f:,.0f}")
    print(f"  Alocação guiada por LINEAR -> valor real: {val_l:,.0f}")
    print(f"  Ganho do fuzzy: {(val_f/val_l-1)*100:+.1f}% | atribuições diferentes: {difere}/{prob_f.N}")

    df = pd.DataFrame([
        {"Modelo de aptidão": "Fuzzy (32 regras)", "Valor real (R$)": round(val_f), "Difere do fuzzy": 0},
        {"Modelo de aptidão": "Linear (surrogate)", "Valor real (R$)": round(val_l), "Difere do fuzzy": difere},
    ])
    df.to_csv(DATA / "exp4_fuzzy_vs_linear.csv", index=False)

    # Onde cada um aloca horas, por classe de saúde do cliente?
    det_f = pd.DataFrame(prob_f.evaluate(ga_f.best_genome, detalhado=True).detalhe)
    det_l = pd.DataFrame(prob_f.evaluate(ga_l.best_genome, detalhado=True).detalhe)
    classes = ["Crítico", "Em Risco", "Atenção", "Saudável", "Promotor"]

    def horas_por_classe(det):
        ativo = det[det["acao"] != "Nenhuma ação"]
        return [ativo[ativo["classe_atual"] == c]["horas"].sum() for c in classes]

    hf, hl = horas_por_classe(det_f), horas_por_classe(det_l)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.bar(["Fuzzy", "Linear"], [val_f, val_l], color=["#2ca02c", "#d62728"])
    ax1.set_ylabel("Valor real no objetivo fuzzy (R$)")
    ax1.set_title("Qualidade da alocação (objetivo verdadeiro)")
    for i, v in enumerate([val_f, val_l]):
        ax1.text(i, v, f"  {v:,.0f}", ha="center", va="bottom")

    x = np.arange(len(classes)); w = 0.38
    ax2.bar(x - w / 2, hf, w, label="Fuzzy", color="#2ca02c")
    ax2.bar(x + w / 2, hl, w, label="Linear", color="#d62728")
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=15)
    ax2.set_ylabel("Horas de CS alocadas")
    ax2.set_title("Onde cada modelo investe as horas")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ASSETS / "exp4_fuzzy_vs_linear.png", dpi=150)
    plt.close(fig)
    print(f"  -> {ASSETS / 'exp4_fuzzy_vs_linear.png'}")


EXPERIMENTS = {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4}

if __name__ == "__main__":
    alvo = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if alvo == "all":
        for fn in EXPERIMENTS.values():
            fn()
            print()
    else:
        EXPERIMENTS[alvo]()
    print(f"Concluído em {time.time()-t0:.0f}s")
