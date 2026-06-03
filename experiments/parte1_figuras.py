"""
Gera todas as figuras e tabelas da Parte 1 (HealthSense — fuzzy CHS).

Saídas:
    assets/mf_plots/*.png         funções de pertinência das 6 variáveis
    assets/control_surfaces/*.png superfícies de controle
    assets/sensibilidade.png      análise de sensibilidade (trilha de ampliação)
    data/cenarios_resultados.csv  tabela dos 14 cenários

Rodar:
    python experiments/gerar_figuras.py
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fuzzy.fuzzy_engine import HealthSenseEngine
from src.fuzzy.membership_functions import build_all_variables, MFParams

MF_DIR = ROOT / "assets" / "parte1" / "mf_plots"
CS_DIR = ROOT / "assets" / "parte1" / "control_surfaces"
DATA = ROOT / "data"
for d in (MF_DIR, CS_DIR, DATA):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. FUNÇÕES DE PERTINÊNCIA
# ============================================================================
def gerar_mfs():
    print("[1] Funções de pertinência...")
    variables = build_all_variables(MFParams())
    titulos = {
        "engajamento": "Engajamento (0-100)",
        "volume_suporte": "Volume de suporte (tickets/mês)",
        "satisfacao": "Satisfação NPS (0-10)",
        "saude_financeira": "Saúde financeira (dias de atraso)",
        "tenure": "Tenure (meses de contrato)",
        "chs": "Customer Health Score (0-100) — saída",
    }
    for nome, var in variables.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for termo, t in var.terms.items():
            ax.plot(var.universe, t.mf, linewidth=2, label=termo)
        ax.set_title(titulos.get(nome, nome))
        ax.set_xlabel(nome)
        ax.set_ylabel("Grau de pertinência")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(MF_DIR / f"{nome}.png", dpi=150)
        plt.close(fig)
    print(f"    -> {MF_DIR} ({len(variables)} figuras)")


# ============================================================================
# 2. SUPERFÍCIES DE CONTROLE
# ============================================================================
def gerar_superficies():
    print("[2] Superfícies de controle...")
    engine = HealthSenseEngine()
    combos = [
        ("engajamento", "satisfacao", {"volume_suporte": 4, "tenure": 18, "saude_financeira": 5}),
        ("saude_financeira", "engajamento", {"volume_suporte": 4, "satisfacao": 7, "tenure": 18}),
        ("volume_suporte", "satisfacao", {"engajamento": 60, "saude_financeira": 5, "tenure": 18}),
        ("tenure", "engajamento", {"volume_suporte": 4, "satisfacao": 7, "saude_financeira": 5}),
    ]
    for x_var, y_var, fixed in combos:
        X, Y, Z = engine.control_surface(x_var, y_var, fixed, resolution=20)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_zlabel("CHS")
        ax.set_title(f"Superfície de controle: {y_var} × {x_var}")
        fig.colorbar(surf, shrink=0.6, label="CHS")
        fig.tight_layout()
        fig.savefig(CS_DIR / f"{y_var}_x_{x_var}.png", dpi=150)
        plt.close(fig)
    print(f"    -> {CS_DIR} ({len(combos)} figuras)")


# ============================================================================
# 3. TABELA DE CENÁRIOS
# ============================================================================
def gerar_cenarios():
    print("[3] Tabela de cenários...")
    engine = HealthSenseEngine()
    cenarios = [
        ("C01", "Crítico — abandonado e inadimplente", 5, 2, 3, 60, 18),
        ("C02", "Crítico — explosão de tickets", 15, 25, 2, 10, 24),
        ("C03", "Crítico — veterano que parou", 10, 5, 4, 20, 48),
        ("C04", "Atenção — novo em onboarding", 45, 6, 7, 0, 2),
        ("C05", "Atenção — uso estagnado", 42, 4, 7, 8, 15),
        ("C06", "Atenção — engajado mas detrator", 75, 3, 3, 0, 20),
        ("C07", "Promotor — power user ideal", 95, 1, 10, 0, 42),
        ("C08", "Saudável — conta normal", 72, 3, 8, 2, 24),
        ("C09", "Promotor — recente animado", 88, 2, 10, 0, 6),
        ("C10", "Fronteira — atenção/saudável", 60, 5, 7, 5, 18),
        ("C11", "Fronteira — risco/atenção", 35, 8, 6, 15, 12),
        ("C12", "Fronteira — saudável/promotor", 82, 2, 9, 0, 30),
        ("C13", "Conflito — power user inadimplente", 92, 1, 8, 75, 24),
        ("C14", "Conflito — promotor com suporte crítico", 70, 22, 10, 0, 18),
    ]
    rows = []
    for cid, desc, eng, sup, sat, fin, ten in cenarios:
        r = engine.evaluate(eng, sup, sat, fin, ten)
        rows.append({
            "ID": cid, "Cenário": desc,
            "Engaj.": eng, "Suporte": sup, "Satisf.": sat, "Atraso": fin, "Tenure": ten,
            "CHS": round(r.chs, 1), "Classificação": r.classificacao,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "cenarios_resultados.csv", index=False)
    print(df[["ID", "CHS", "Classificação"]].to_string(index=False))
    print(f"    -> {DATA / 'cenarios_resultados.csv'}")


# ============================================================================
# 4. ANÁLISE DE SENSIBILIDADE (trilha de ampliação)
# ============================================================================
def gerar_sensibilidade():
    print("[4] Análise de sensibilidade...")
    rng = np.random.default_rng(42)
    personas = pd.DataFrame({
        "engajamento": rng.uniform(0, 100, 100),
        "volume_suporte": rng.uniform(0, 30, 100),
        "satisfacao": rng.uniform(0, 10, 100),
        "saude_financeira": rng.uniform(0, 90, 100),
        "tenure": rng.uniform(0, 60, 100),
    })

    def chs_medio(params):
        eng = HealthSenseEngine(params)
        return float(np.mean([eng.evaluate(*r).chs for r in personas.itertuples(index=False)]))

    base = MFParams()
    centros = [60, 65, 70, 75, 80]
    medios = []
    for centro in centros:
        delta = centro - 70
        novo = replace(base, engajamento_alto=(55 + delta, centro, 85 + delta))
        medios.append(chs_medio(novo))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centros, medios, marker="o", linewidth=2)
    ax.set_xlabel("Centro da MF 'engajamento_alto'")
    ax.set_ylabel("CHS médio (100 personas)")
    ax.set_title("Análise de sensibilidade — MF engajamento_alto")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "parte1" / "sensibilidade.png", dpi=150)
    plt.close(fig)
    for c, m in zip(centros, medios):
        print(f"    centro={c} -> CHS médio={m:.2f}")
    print(f"    -> {ROOT / 'assets' / 'parte1' / 'sensibilidade.png'}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    gerar_mfs()
    gerar_superficies()
    gerar_cenarios()
    gerar_sensibilidade()
    print(f"Concluído em {time.time()-t0:.0f}s")
