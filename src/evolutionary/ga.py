"""
Algoritmo Genético (AG) para o HealthSense Allocator.

Componentes do motor evolutivo (exigidos pela rubrica):
  - Representação .......... vetor de inteiros (atribuição cliente → opção)
  - Função de aptidão ...... problem.fitness (receita esperada penalizada)
  - Seleção ................ torneio (tournament_size)
  - Crossover .............. uniforme (por gene)
  - Mutação ................ reatribuição por gene (mutation_rate)
  - Elitismo ............... preserva os melhores indivíduos
  - Critério de parada ..... nº de gerações OU estagnação (patience)

Suporta um gancho opcional de BUSCA LOCAL (`local_search`), usado pela versão
Memética (memetic.py) — assim AG e Memético compartilham o mesmo núcleo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .problem import AllocationProblem, FitnessResult


# ============================================================================
# CONFIGURAÇÃO E RESULTADO
# ============================================================================
@dataclass
class GAConfig:
    pop_size: int = 80
    n_generations: int = 120
    crossover_rate: float = 0.9
    mutation_rate: float = 0.03          # prob. por gene
    tournament_size: int = 3
    elitism: int = 2
    stagnation_patience: int = 30        # gerações sem melhora -> para
    ls_elites: int = 0                   # memético: nº de elites refinados por busca local
    seed: int = 0


@dataclass
class GAResult:
    best_genome: np.ndarray
    best_fitness: float
    best_result: FitnessResult
    history_best: List[float] = field(default_factory=list)
    history_mean: List[float] = field(default_factory=list)
    history_evals: List[int] = field(default_factory=list)  # avaliações acumuladas por geração
    generations: int = 0
    evaluations: int = 0
    wall_time: float = 0.0
    algoritmo: str = "AG"


# ============================================================================
# ALGORITMO GENÉTICO
# ============================================================================
class GeneticAlgorithm:
    def __init__(
        self,
        problem: AllocationProblem,
        config: Optional[GAConfig] = None,
        local_search: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        use_repair: bool = True,
        seeds: Optional[List[np.ndarray]] = None,
    ):
        self.problem = problem
        self.cfg = config or GAConfig()
        self.local_search = local_search
        self.use_repair = use_repair
        self.seeds = seeds or []
        self.rng = np.random.default_rng(self.cfg.seed)
        self.evaluations = 0

    def _maybe_repair(self, genome: np.ndarray) -> np.ndarray:
        return self.problem.repair(genome) if self.use_repair else genome

    # ------------------------------------------------------------------------
    # EXECUÇÃO PRINCIPAL
    # ------------------------------------------------------------------------
    def run(self) -> GAResult:
        import time
        t_start = time.time()
        cfg = self.cfg
        # População inicial: sementes fornecidas (ex.: solução gulosa) + aleatórios,
        # todos passados pelo reparo para garantir factibilidade desde o início.
        population = [self._maybe_repair(np.asarray(s, dtype=int)) for s in self.seeds]
        while len(population) < cfg.pop_size:
            population.append(self._maybe_repair(self.problem.random_genome(self.rng)))
        population = population[: cfg.pop_size]
        fitnesses = np.array([self._fit(g) for g in population])

        history_best: List[float] = []
        history_mean: List[float] = []
        history_evals: List[int] = []
        best_idx = int(np.argmax(fitnesses))
        best_genome = population[best_idx].copy()
        best_fit = float(fitnesses[best_idx])
        gens_sem_melhora = 0
        gen = 0

        for gen in range(1, cfg.n_generations + 1):
            # --- Elitismo: preserva os melhores ---
            ordem = np.argsort(fitnesses)[::-1]
            nova_pop = [population[i].copy() for i in ordem[: cfg.elitism]]
            novas_fit = [float(fitnesses[i]) for i in ordem[: cfg.elitism]]

            # --- Gera descendentes até completar a população ---
            while len(nova_pop) < cfg.pop_size:
                pai1 = self._selecao_torneio(population, fitnesses)
                pai2 = self._selecao_torneio(population, fitnesses)
                filho = self._crossover(pai1, pai2)
                filho = self._mutacao(filho)
                filho = self._maybe_repair(filho)
                nova_pop.append(filho)
                novas_fit.append(self._fit(filho))

            population = nova_pop
            fitnesses = np.array(novas_fit)

            # --- Memético: refina a elite com busca local (aprendizado lamarckiano) ---
            if self.local_search is not None and cfg.ls_elites > 0:
                top = np.argsort(fitnesses)[::-1][: cfg.ls_elites]
                for k in top:
                    refinado = self.local_search(population[k])
                    f_ref = self._fit(refinado)
                    if f_ref > fitnesses[k]:
                        population[k] = refinado
                        fitnesses[k] = f_ref

            # --- Atualiza melhor global ---
            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_fit + 1e-9:
                best_fit = float(fitnesses[gen_best_idx])
                best_genome = population[gen_best_idx].copy()
                gens_sem_melhora = 0
            else:
                gens_sem_melhora += 1

            history_best.append(best_fit)
            history_mean.append(float(np.mean(fitnesses)))
            history_evals.append(self.evaluations)

            # --- Critério de parada por estagnação ---
            if gens_sem_melhora >= cfg.stagnation_patience:
                break

        return GAResult(
            best_genome=best_genome,
            best_fitness=best_fit,
            best_result=self.problem.evaluate(best_genome, detalhado=True),
            history_best=history_best,
            history_mean=history_mean,
            history_evals=history_evals,
            generations=gen,
            evaluations=self.evaluations,
            wall_time=time.time() - t_start,
            algoritmo="Memético" if self.local_search is not None else "AG",
        )

    # ------------------------------------------------------------------------
    # OPERADORES
    # ------------------------------------------------------------------------
    def _selecao_torneio(self, population: List[np.ndarray], fitnesses: np.ndarray) -> np.ndarray:
        """Seleção por torneio: amostra k indivíduos e devolve o melhor."""
        idxs = self.rng.integers(0, len(population), size=self.cfg.tournament_size)
        melhor = idxs[int(np.argmax(fitnesses[idxs]))]
        return population[melhor]

    def _crossover(self, pai1: np.ndarray, pai2: np.ndarray) -> np.ndarray:
        """Crossover uniforme: cada gene vem de um dos pais com prob. 0,5.

        Adequado para vetores de atribuição (cada posição é independente).
        """
        if self.rng.random() > self.cfg.crossover_rate:
            return pai1.copy()
        mascara = self.rng.random(self.problem.N) < 0.5
        filho = np.where(mascara, pai1, pai2)
        return filho.astype(int)

    def _mutacao(self, genome: np.ndarray) -> np.ndarray:
        """Mutação por gene: cada posição é reatribuída com prob. mutation_rate."""
        genome = genome.copy()
        mascara = self.rng.random(self.problem.N) < self.cfg.mutation_rate
        n_mut = int(mascara.sum())
        if n_mut:
            genome[mascara] = self.rng.integers(0, self.problem.n_options, size=n_mut)
        return genome

    # ------------------------------------------------------------------------
    # INTERNO
    # ------------------------------------------------------------------------
    def _fit(self, genome: np.ndarray) -> float:
        self.evaluations += 1
        return self.problem.fitness(genome)


def run_ga(problem: AllocationProblem, seed: int = 0, **kwargs) -> GAResult:
    """Atalho de conveniência para rodar o AG com uma semente."""
    cfg = GAConfig(seed=seed, **kwargs)
    return GeneticAlgorithm(problem, cfg).run()
