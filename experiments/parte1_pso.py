"""
Parte 1 — Pontuação extra: otimização dos parâmetros das funções de pertinência
do Customer Health Score via PSO (Particle Swarm Optimization).

Função objetivo: minimizar o erro médio absoluto entre o CHS do sistema e o CHS
rotulado por "especialista" em 10 personas. Otimiza os centros das gaussianas da
saída (5 parâmetros). Gera comparação antes/depois e curva de convergência.

Rodar:
    python experiments/parte1_pso.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyswarms as ps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fuzzy.fuzzy_engine import HealthSenseEngine
from src.fuzzy.membership_functions import MFParams

ASSETS = ROOT / "assets" / "parte1"
ASSETS.mkdir(parents=True, exist_ok=True)

# Dataset rotulado por "especialista": (engaj, sup, sat, fin, ten, chs_esperado)
DATASET = pd.DataFrame([
    (95, 1, 10, 0,  42, 92),
    (88, 2, 10, 0,  6,  82),
    (72, 3, 8,  2,  24, 70),
    (60, 5, 7,  5,  18, 60),
    (45, 6, 7,  0,  2,  50),
    (42, 4, 7,  8,  15, 48),
    (35, 8, 6,  15, 12, 38),
    (15, 25, 2, 10, 24, 12),
    (5,  2,  3, 60, 18, 10),
    (10, 5,  4, 20, 48, 18),
], columns=["engajamento", "volume_suporte", "satisfacao", "saude_financeira", "tenure", "chs_esperado"])

BASE = MFParams()


def make_params(x: np.ndarray) -> MFParams:
    """x = centros das gaussianas: [critico, em_risco, atencao, saudavel, promotor]."""
    return replace(
        BASE,
        chs_critico=(x[0], 8), chs_em_risco=(x[1], 8), chs_atencao=(x[2], 8),
        chs_saudavel=(x[3], 8), chs_promotor=(x[4], 8),
    )


def erro_para(x: np.ndarray) -> float:
    eng = HealthSenseEngine(make_params(x))
    erros = [
        abs(eng.evaluate(r.engajamento, r.volume_suporte, r.satisfacao,
                         r.saude_financeira, r.tenure).chs - r.chs_esperado)
        for r in DATASET.itertuples(index=False)
    ]
    return float(np.mean(erros))


def objetivo(X: np.ndarray) -> np.ndarray:
    return np.array([erro_para(p) for p in X])


def main():
    baseline = np.array([10, 30, 50, 70, 90])
    erro_base = erro_para(baseline)
    print(f"Erro médio absoluto (baseline): {erro_base:.2f}")

    lb = np.array([0, 15, 35, 55, 75])
    ub = np.array([20, 45, 65, 85, 100])
    optimizer = ps.single.GlobalBestPSO(
        n_particles=20, dimensions=5,
        options={"c1": 1.5, "c2": 1.5, "w": 0.7}, bounds=(lb, ub),
    )
    best_cost, best_pos = optimizer.optimize(objetivo, iters=30, verbose=True)
    print(f"\nMelhor erro: {best_cost:.2f} | centros: {best_pos.round(1)}")
    print(f"Redução: {(1 - best_cost / erro_base) * 100:.1f}%")

    # Comparação antes/depois
    eng_b = HealthSenseEngine(make_params(baseline))
    eng_o = HealthSenseEngine(make_params(best_pos))
    rows = []
    for r in DATASET.itertuples(index=False):
        cb = eng_b.evaluate(r.engajamento, r.volume_suporte, r.satisfacao, r.saude_financeira, r.tenure).chs
        co = eng_o.evaluate(r.engajamento, r.volume_suporte, r.satisfacao, r.saude_financeira, r.tenure).chs
        rows.append({"esperado": r.chs_esperado, "baseline": round(cb, 1), "otimizado": round(co, 1),
                     "erro_base": round(abs(cb - r.chs_esperado), 1), "erro_otim": round(abs(co - r.chs_esperado), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "pso_antes_depois.csv", index=False)

    # Convergência
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(optimizer.cost_history, marker="o", markersize=4)
    ax.set_xlabel("Iteração"); ax.set_ylabel("Erro médio absoluto")
    ax.set_title("Convergência do PSO")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "pso_convergencia.png", dpi=150)
    plt.close(fig)

    # Antes/depois por persona
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df)); w = 0.35
    ax.bar(x - w / 2, df["erro_base"], w, label="Baseline", color="salmon")
    ax.bar(x + w / 2, df["erro_otim"], w, label="Otimizado (PSO)", color="seagreen")
    ax.set_xlabel("Persona"); ax.set_ylabel("Erro absoluto |CHS - esperado|")
    ax.set_title("Erro por persona: baseline vs. otimizado por PSO")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ASSETS / "pso_antes_depois.png", dpi=150)
    plt.close(fig)
    print(f"Figuras -> {ASSETS}")


if __name__ == "__main__":
    main()
