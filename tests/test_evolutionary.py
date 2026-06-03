"""
Testes do motor evolutivo do HealthSense Allocator.

Cobrem: construção do problema, factibilidade do reparo, propriedades dos
baselines e a propriedade central de validação — o AG supera a busca aleatória
e iguala/supera a heurística gulosa, sempre respeitando as restrições.

Usa uma instância pequena (N=24) para manter os testes rápidos (~poucos seg).

Rodar:
    pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evolutionary.problem import make_default_problem
from src.evolutionary.ga import GeneticAlgorithm, GAConfig, run_ga
from src.evolutionary.memetic import run_memetic
from src.evolutionary import baselines


@pytest.fixture(scope="module")
def problem():
    # Instância pequena para velocidade; semente fixa para reprodutibilidade.
    return make_default_problem(n_customers=24, seed=7)


# ============================================================================
# CONSTRUÇÃO E DECODIFICAÇÃO
# ============================================================================
def test_problema_construido(problem):
    # N é aproximado (arredondamento das frações dos arquétipos) — ~24
    assert 20 <= problem.N <= 28
    assert problem.M == 5
    assert problem.K == 5
    assert problem.n_options == problem.M * problem.K + 1


def test_decode_roundtrip(problem):
    # gene 0 = nenhuma ação
    assert problem.decode(0) == (None, None)
    # genes válidos decodificam para (csm, playbook) dentro dos limites
    for gene in range(1, problem.n_options):
        csm, pb = problem.decode(gene)
        assert 0 <= csm < problem.M
        assert 0 <= pb < problem.K


def test_capacidade_menor_que_demanda(problem):
    """O problema deve ser de escassez: não há horas para dar a TODO cliente o
    playbook mais caro — força o sistema a escolher quem atender e como rotear."""
    max_horas = max(p.custo_horas for p in problem.playbooks)
    assert problem.capacidade_total < problem.N * max_horas


# ============================================================================
# REPARO E FACTIBILIDADE
# ============================================================================
def test_reparo_produz_factivel(problem):
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = problem.random_genome(rng)
        reparado = problem.repair(g)
        assert problem.evaluate(reparado).factivel, "reparo deveria garantir factibilidade"


def test_avaliacao_dentro_dos_limites(problem):
    """Valor total nunca negativo; horas por CSM consistentes."""
    rng = np.random.default_rng(1)
    g = problem.repair(problem.random_genome(rng))
    res = problem.evaluate(g)
    assert res.valor_total >= 0
    assert res.horas_excedidas <= 1e-9


# ============================================================================
# BASELINES
# ============================================================================
def test_no_action_zero(problem):
    res = baselines.no_action(problem)
    assert res.best_fitness == pytest.approx(0.0)
    assert res.best_result.n_atendidos == 0


def test_greedy_factivel(problem):
    res = baselines.greedy(problem)
    assert res.best_result.factivel
    assert res.best_fitness > 0


def test_greedy_supera_aleatorio(problem):
    g = baselines.greedy(problem)
    r = baselines.random_search(problem, n_samples=2000, seed=0)
    assert g.best_fitness > r.best_fitness


# ============================================================================
# ALGORITMO GENÉTICO — propriedades centrais de validação
# ============================================================================
def test_ga_factivel_e_positivo(problem):
    res = run_ga(problem, seed=0, pop_size=60, n_generations=80)
    assert res.best_result.factivel
    assert res.best_fitness > 0


def test_ga_supera_aleatorio(problem):
    ga = run_ga(problem, seed=0, pop_size=60, n_generations=80)
    rand = baselines.random_search(problem, n_samples=4800, seed=0)
    assert ga.best_fitness > rand.best_fitness * 1.10  # margem folgada


def test_ga_iguala_ou_supera_guloso(problem):
    """Propriedade-chave: o AG não fica abaixo do guloso (com folga de ruído)."""
    greedy = baselines.greedy(problem)
    ga = run_ga(problem, seed=0, pop_size=80, n_generations=120)
    assert ga.best_fitness >= greedy.best_fitness * 0.98


def test_ga_deterministico_por_semente(problem):
    a = run_ga(problem, seed=42, pop_size=50, n_generations=40)
    b = run_ga(problem, seed=42, pop_size=50, n_generations=40)
    assert a.best_fitness == pytest.approx(b.best_fitness)


def test_ga_conta_avaliacoes(problem):
    res = run_ga(problem, seed=0, pop_size=40, n_generations=30)
    assert res.evaluations > 0
    assert len(res.history_best) == res.generations


# ============================================================================
# MEMÉTICO
# ============================================================================
def test_memetico_factivel(problem):
    res = run_memetic(problem, seed=0, pop_size=50, n_generations=40)
    assert res.best_result.factivel
    assert res.best_fitness > 0


def test_memetico_conta_avaliacoes_da_busca_local(problem):
    """As avaliações da busca local DEVEM ser contabilizadas (exigência da rubrica).

    O memético, por fazer busca local, deve gastar MAIS avaliações que o AG puro
    no mesmo nº de gerações.
    """
    ga = run_ga(problem, seed=0, pop_size=50, n_generations=40)
    mem = run_memetic(problem, seed=0, pop_size=50, n_generations=40)
    assert mem.evaluations > ga.evaluations


# ============================================================================
# INTEGRAÇÃO FUZZY-EVOLUTIVA (pontuação extra)
# ============================================================================
def test_fuzzy_vs_linear_diferem():
    """Alocação guiada por fuzzy difere da guiada por linear e é melhor no
    objetivo verdadeiro (fuzzy). Garante que o fuzzy não é decorativo."""
    from src.fuzzy import HealthSenseEngine
    eng = HealthSenseEngine()
    prob_f = make_default_problem(n_customers=24, seed=7, fitness_mode="fuzzy", engine=eng)
    prob_l = make_default_problem(n_customers=24, seed=7, fitness_mode="linear", engine=eng)
    ga_f = run_ga(prob_f, seed=0, pop_size=60, n_generations=100)
    ga_l = run_ga(prob_l, seed=0, pop_size=60, n_generations=100)

    # avaliadas sob o objetivo VERDADEIRO (fuzzy)
    val_f = prob_f.evaluate(ga_f.best_genome).valor_total
    val_l = prob_f.evaluate(ga_l.best_genome).valor_total

    assert np.sum(ga_f.best_genome != ga_l.best_genome) > 0, "alocações deveriam diferir"
    assert val_f > val_l, "fuzzy deveria render mais no objetivo verdadeiro"
