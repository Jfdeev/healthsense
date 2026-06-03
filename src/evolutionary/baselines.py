"""
Métodos de comparação (baselines) — exigência da rubrica:
"comparar o resultado com solução manual, busca aleatória, heurística gulosa,
versão sem otimização ou método alternativo simples".

  1. no_action ....... versão SEM otimização (ninguém é atendido) — piso.
  2. random_search ... busca aleatória com orçamento de avaliações igual ao do AG.
  3. greedy .......... heurística gulosa por densidade de valor (valor/hora).
                       É o baseline FORTE que o AG precisa superar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .problem import AllocationProblem, FitnessResult


@dataclass
class BaselineResult:
    best_genome: np.ndarray
    best_fitness: float
    best_result: FitnessResult
    evaluations: int
    algoritmo: str
    history_best: List[float] = field(default_factory=list)


# ============================================================================
# 1. SEM OTIMIZAÇÃO
# ============================================================================
def no_action(problem: AllocationProblem) -> BaselineResult:
    """Ninguém é atendido. Receita esperada = 0. É o piso de comparação."""
    genome = np.zeros(problem.N, dtype=int)
    res = problem.evaluate(genome, detalhado=True)
    return BaselineResult(genome, res.fitness, res, 1, "Sem otimização")


# ============================================================================
# 2. BUSCA ALEATÓRIA
# ============================================================================
def random_search(problem: AllocationProblem, n_samples: int = 9600, seed: int = 0) -> BaselineResult:
    """Amostra n_samples genomas aleatórios e devolve o melhor.

    n_samples deve ser comparável ao nº de avaliações do AG (pop × gerações)
    para que a comparação seja justa (mesmo orçamento computacional).
    """
    rng = np.random.default_rng(seed)
    best_genome = problem.repair(problem.random_genome(rng))
    best_fit = problem.fitness(best_genome)
    history = [best_fit]

    for _ in range(n_samples - 1):
        g = problem.repair(problem.random_genome(rng))
        f = problem.fitness(g)
        if f > best_fit:
            best_fit, best_genome = f, g
        history.append(best_fit)

    return BaselineResult(
        best_genome, best_fit, problem.evaluate(best_genome, detalhado=True),
        n_samples, "Busca aleatória", history,
    )


# ============================================================================
# 3. HEURÍSTICA GULOSA (densidade de valor)
# ============================================================================
def greedy(problem: AllocationProblem) -> BaselineResult:
    """Atribui ações por ordem decrescente de densidade de valor (valor/hora),
    respeitando o orçamento de cada CSM e uma ação por cliente.

    Heurística que um gerente de CS faria "na mão": priorizar as intervenções
    de maior retorno por hora investida. É boa, mas é gulosa — não enxerga
    o acoplamento global das capacidades, por isso é superável pelo AG.
    """
    # Monta candidatos (cliente, gene, valor, horas, csm, densidade)
    candidatos = []
    for i in range(problem.N):
        for gene in range(1, problem.n_options):
            valor = problem._valor[i, gene]
            horas = problem._horas[i, gene]
            if valor <= 0 or horas <= 0:
                continue
            csm_idx = int(problem._csm_de[i, gene])
            densidade = valor / horas
            candidatos.append((densidade, valor, horas, i, gene, csm_idx))

    # Ordena por densidade decrescente (desempata por valor)
    candidatos.sort(key=lambda x: (x[0], x[1]), reverse=True)

    genome = np.zeros(problem.N, dtype=int)
    atribuido = np.zeros(problem.N, dtype=bool)
    orcamento_rest = np.array([c.orcamento_horas for c in problem.csms], dtype=float)

    for densidade, valor, horas, i, gene, csm_idx in candidatos:
        if atribuido[i]:
            continue
        if orcamento_rest[csm_idx] + 1e-9 >= horas:
            genome[i] = gene
            atribuido[i] = True
            orcamento_rest[csm_idx] -= horas

    return BaselineResult(
        genome, problem.fitness(genome), problem.evaluate(genome, detalhado=True),
        len(candidatos), "Guloso (densidade)",
    )
