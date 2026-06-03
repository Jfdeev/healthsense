"""
Algoritmo Memético (AM) = Algoritmo Genético + Busca Local.

É a versão HÍBRIDA usada na "Comparação ampliada" (ampliação obrigatória de
equipes de 5 integrantes): comparar versão simples (AG) vs versão híbrida (AM).

A busca local é um hill-climbing de primeira-melhora: tenta trocar o playbook/CSM
de um cliente por vez; se melhora a aptidão (já com reparo de factibilidade),
aceita a troca. Aplicada aos melhores indivíduos a cada geração — o que o
algoritmo memético chama de "aprendizado individual" (refinamento lamarckiano).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .ga import GAConfig, GAResult, GeneticAlgorithm
from .problem import AllocationProblem


def make_local_search(
    problem: AllocationProblem,
    rng: np.random.Generator,
    fitness_fn=None,
    max_passes: int = 2,
    sample_genes: Optional[int] = None,
    first_improvement: bool = False,
):
    """Cria a função de busca local (hill-climbing) acoplada ao problema.

    fitness_fn: função de aptidão a usar. DEVE ser o contador do algoritmo
                (self._fit) para que as avaliações da busca local sejam
                contabilizadas — a rubrica exige "número de avaliações da função
                objetivo" como métrica, e o memético faz muitas na busca local.
    max_passes: nº de varreduras completas sobre os clientes.
    sample_genes: se definido, amostra só esse nº de clientes por passe (mais
                  barato). Se None, varre todos.
    """
    fit = fitness_fn or problem.fitness

    def local_search(genome: np.ndarray) -> np.ndarray:
        cur = problem.repair(np.asarray(genome, dtype=int).copy())
        best_fit = fit(cur)

        for _ in range(max_passes):
            melhorou = False
            indices = np.arange(problem.N)
            if sample_genes is not None and sample_genes < problem.N:
                indices = rng.choice(problem.N, size=sample_genes, replace=False)

            for i in indices:
                gene_atual = int(cur[i])
                melhor_variante = cur
                melhor_fit_local = best_fit
                # Testa cada opção para o cliente i sobre uma CÓPIA isolada
                for novo_gene in range(problem.n_options):
                    if novo_gene == gene_atual:
                        continue
                    trial = cur.copy()
                    trial[i] = novo_gene
                    trial = problem.repair(trial)
                    f = fit(trial)
                    if f > melhor_fit_local + 1e-9:
                        melhor_fit_local = f
                        melhor_variante = trial
                        if first_improvement:
                            break  # aceita a 1ª melhora (mais barato)
                # commit da melhor troca encontrada para o cliente i
                if melhor_fit_local > best_fit + 1e-9:
                    cur = melhor_variante
                    best_fit = melhor_fit_local
                    melhorou = True
            if not melhorou:
                break
        return cur

    return local_search


class MemeticAlgorithm(GeneticAlgorithm):
    """AG com busca local plugada (refinamento dos descendentes)."""

    def __init__(
        self,
        problem: AllocationProblem,
        config: Optional[GAConfig] = None,
        seeds=None,
        ls_max_passes: int = 1,
        ls_sample_genes: Optional[int] = 12,
        ls_first_improvement: bool = False,
    ):
        cfg = config or GAConfig()
        if cfg.ls_elites <= 0:
            cfg.ls_elites = 3  # refina os 3 melhores por geração
        # Inicializa o GA primeiro (cria self._fit / contador), depois pluga a busca
        # local roteando as avaliações pelo contador (self._fit).
        super().__init__(problem, cfg, local_search=None, seeds=seeds)
        rng_ls = np.random.default_rng(cfg.seed + 999)
        self.local_search = make_local_search(
            problem, rng_ls, fitness_fn=self._fit,
            max_passes=ls_max_passes, sample_genes=ls_sample_genes,
            first_improvement=ls_first_improvement,
        )

    def run(self) -> GAResult:
        res = super().run()
        res.algoritmo = "Memético"
        return res


def run_memetic(problem: AllocationProblem, seed: int = 0, seeds=None, **kwargs) -> GAResult:
    """Atalho para rodar o Memético com uma semente."""
    cfg = GAConfig(seed=seed, **kwargs)
    return MemeticAlgorithm(problem, cfg, seeds=seeds).run()
