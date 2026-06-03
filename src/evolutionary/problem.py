"""
HealthSense Allocator — Formulação do problema de otimização.

PROBLEMA (Generalized Assignment Problem — GAP):
    Um time de Customer Success tem capacidade limitada. Para cada cliente da
    carteira, decidir QUAL CSM o atende com QUAL playbook (ou nenhuma ação),
    maximizando a receita esperada retida/expandida, respeitando o orçamento
    de horas de CADA CSM.

POR QUE É NP-DIFÍCIL (e não trivial):
    - Múltiplas restrições de capacidade (uma por CSM), não uma só.
    - O valor de uma ação depende do trio (cliente, CSM, playbook), porque cada
      CSM tem uma ESPECIALIDADE que escala o efeito do playbook.
    - A função objetivo é NÃO-LINEAR: o ganho de saúde (ΔCHS) é calculado pelo
      motor fuzzy Mamdani da Parte 1 (rule-based, operadores min/max), não por
      uma fórmula linear. Isso impede solução exata por programação linear.

    Essa combinação (atribuição sob múltiplas capacidades + objetivo fuzzy
    não-linear) é exatamente o terreno onde um Algoritmo Genético é adequado.

FUNÇÃO DE FITNESS (integração Fuzzy-Evolutiva — pontuação extra):
    fitness(x) = Σ_i  MRR_i × (ΔCHS_i / 100)  −  penalidade × horas_excedidas
    onde ΔCHS_i = CHS_fuzzy(estado_pós_ação) − CHS_fuzzy(estado_atual).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.fuzzy import HealthSenseEngine, classify_chs


# ============================================================================
# ENTIDADES DO DOMÍNIO
# ============================================================================
INPUT_VARS = ("engajamento", "volume_suporte", "satisfacao", "saude_financeira", "tenure")
VAR_BOUNDS = {
    "engajamento": (0, 100),
    "volume_suporte": (0, 30),
    "satisfacao": (0, 10),
    "saude_financeira": (0, 90),
    "tenure": (0, 60),
}


@dataclass
class Customer:
    """Conta da carteira de SaaS B2B."""
    cid: str
    nome: str
    engajamento: float
    volume_suporte: float
    satisfacao: float
    saude_financeira: float
    tenure: float
    mrr: float  # Monthly Recurring Revenue (R$/mês)

    def state(self) -> dict:
        return {v: getattr(self, v) for v in INPUT_VARS}


@dataclass
class Playbook:
    """Intervenção de Customer Success com custo em horas e efeito nas variáveis."""
    pid: str
    nome: str
    custo_horas: float
    efeito: Dict[str, float] = field(default_factory=dict)  # var -> delta


@dataclass
class CSM:
    """Customer Success Manager: tem orçamento de horas e uma especialidade."""
    sid: str
    nome: str
    orcamento_horas: float
    especialidade: str  # "Onboarding" | "Retenção" | "Adoção"


# ============================================================================
# RESULTADO DE UMA AVALIAÇÃO
# ============================================================================
@dataclass
class FitnessResult:
    fitness: float                 # objetivo penalizado (o que o GA maximiza)
    valor_total: float             # receita esperada retida/expandida (R$)
    horas_por_csm: Dict[str, float]
    horas_excedidas: float
    factivel: bool
    n_atendidos: int
    detalhe: List[dict] = field(default_factory=list)  # por cliente (para relatório)


# ============================================================================
# PROBLEMA DE ALOCAÇÃO
# ============================================================================
class AllocationProblem:
    """Encapsula carteira, time de CS, playbooks e a função de fitness.

    Codificação da solução (genoma):
        Vetor de inteiros de tamanho N (n_customers). Cada gene g_i ∈ {0..n_opt-1}:
            g_i == 0           -> nenhuma ação para o cliente i
            g_i >= 1           -> decodifica em (csm, playbook):
                                   csm      = (g_i - 1) // K
                                   playbook = (g_i - 1)  % K
        onde K = número de playbooks reais (sem contar "nenhuma ação").
    """

    PENALTY_COEF = 1000.0  # R$ de penalidade por hora excedida (>> densidade de valor)

    def __init__(
        self,
        customers: List[Customer],
        csms: List[CSM],
        playbooks: List[Playbook],
        specialty_mult: Dict[Tuple[str, str], float],
        engine: Optional[HealthSenseEngine] = None,
        fitness_mode: str = "fuzzy",  # "fuzzy" | "linear" (para análise de impacto)
    ):
        self.customers = customers
        self.csms = csms
        self.playbooks = playbooks  # apenas os REAIS (sem no-action)
        self.specialty_mult = specialty_mult
        self.engine = engine or HealthSenseEngine()
        self.fitness_mode = fitness_mode

        self.N = len(customers)
        self.M = len(csms)
        self.K = len(playbooks)
        self.n_options = self.M * self.K + 1  # +1 = nenhuma ação

        # CHS atual de cada cliente (uma avaliação fuzzy por cliente)
        self._chs_atual = np.array([
            self.engine.evaluate(**c.state()).chs for c in customers
        ])

        # Pré-computa a tabela de valor/horas/csm de CADA opção para CADA cliente.
        # Custo: N × n_options avaliações fuzzy, feito UMA vez. Depois, avaliar um
        # genoma é só lookup + contabilidade de capacidade (rápido).
        # Isso NÃO trivializa o problema: a dificuldade está em satisfazer as M
        # restrições de capacidade simultâneas (GAP é NP-difícil mesmo com valores
        # tabelados).
        self._valor = np.zeros((self.N, self.n_options))
        self._horas = np.zeros((self.N, self.n_options))
        self._csm_de = np.full((self.N, self.n_options), -1, dtype=int)
        self._delta_chs = np.zeros((self.N, self.n_options))
        self._precompute()

    # ------------------------------------------------------------------------
    # DECODIFICAÇÃO
    # ------------------------------------------------------------------------
    def decode(self, gene: int) -> Tuple[Optional[int], Optional[int]]:
        """gene -> (índice do CSM, índice do playbook) ou (None, None) para no-action."""
        if gene <= 0:
            return None, None
        idx = gene - 1
        return idx // self.K, idx % self.K

    def option_label(self, i: int, gene: int) -> str:
        csm_idx, pb_idx = self.decode(gene)
        if csm_idx is None:
            return "Nenhuma ação"
        return f"{self.csms[csm_idx].nome} → {self.playbooks[pb_idx].nome}"

    # ------------------------------------------------------------------------
    # PRÉ-COMPUTO DA TABELA DE VALOR
    # ------------------------------------------------------------------------
    def _apply_effect(self, customer: Customer, playbook: Playbook, mult: float) -> dict:
        """Aplica o efeito do playbook (escalado pela especialidade) ao estado."""
        estado = customer.state()
        for var, delta in playbook.efeito.items():
            lo, hi = VAR_BOUNDS[var]
            estado[var] = float(np.clip(estado[var] + delta * mult, lo, hi))
        return estado

    def _precompute(self) -> None:
        for i, customer in enumerate(self.customers):
            chs0 = self._chs_atual[i]
            # gene 0 = nenhuma ação: valor 0, sem horas
            for gene in range(1, self.n_options):
                csm_idx, pb_idx = self.decode(gene)
                csm = self.csms[csm_idx]
                playbook = self.playbooks[pb_idx]
                mult = self.specialty_mult.get((csm.especialidade, playbook.pid), 1.0)

                novo_estado = self._apply_effect(customer, playbook, mult)
                chs_novo = self.engine.evaluate(**novo_estado).chs
                delta = chs_novo - chs0

                self._delta_chs[i, gene] = delta
                self._horas[i, gene] = playbook.custo_horas
                self._csm_de[i, gene] = csm_idx

                if self.fitness_mode == "fuzzy":
                    valor = customer.mrr * (delta / 100.0)
                else:
                    # Modo LINEAR (surrogate sem fuzzy): recompensa a magnitude bruta
                    # do efeito aplicado, ignorando a saturação fuzzy. Usado só na
                    # análise de impacto (extra) para mostrar que o fuzzy muda a alocação.
                    magnitude = sum(
                        abs(d) * mult for d in playbook.efeito.values()
                    )
                    valor = customer.mrr * (magnitude / 100.0)
                self._valor[i, gene] = valor

    # ------------------------------------------------------------------------
    # AVALIAÇÃO DE UM GENOMA
    # ------------------------------------------------------------------------
    def evaluate(self, genome: np.ndarray, detalhado: bool = False) -> FitnessResult:
        genome = np.asarray(genome, dtype=int)
        horas_csm = np.zeros(self.M)
        valor_total = 0.0
        n_atendidos = 0
        detalhe: List[dict] = []

        for i, gene in enumerate(genome):
            if gene <= 0:
                if detalhado:
                    detalhe.append(self._linha_detalhe(i, gene))
                continue
            valor_total += self._valor[i, gene]
            csm_idx = self._csm_de[i, gene]
            horas_csm[csm_idx] += self._horas[i, gene]
            n_atendidos += 1
            if detalhado:
                detalhe.append(self._linha_detalhe(i, gene))

        # Penalidade por estourar o orçamento de qualquer CSM
        excedente = np.maximum(0.0, horas_csm - np.array([c.orcamento_horas for c in self.csms]))
        horas_excedidas = float(excedente.sum())
        fitness = valor_total - self.PENALTY_COEF * horas_excedidas

        return FitnessResult(
            fitness=float(fitness),
            valor_total=float(valor_total),
            horas_por_csm={self.csms[j].nome: float(horas_csm[j]) for j in range(self.M)},
            horas_excedidas=horas_excedidas,
            factivel=horas_excedidas <= 1e-9,
            n_atendidos=n_atendidos,
            detalhe=detalhe,
        )

    def fitness(self, genome: np.ndarray) -> float:
        """Atalho que retorna só o escalar de fitness (usado nos algoritmos)."""
        return self.evaluate(genome).fitness

    # ------------------------------------------------------------------------
    # OPERADOR DE REPARO (factibilidade)
    # ------------------------------------------------------------------------
    def repair(self, genome: np.ndarray) -> np.ndarray:
        """Torna o genoma factível: para cada CSM acima do orçamento, remove as
        atribuições de PIOR densidade de valor (valor/hora) até caber.

        Manter a população factível faz o AG otimizar valor diretamente na região
        viável — técnica padrão para problemas de mochila/atribuição, muito mais
        eficaz que confiar só em penalidade.
        """
        genome = np.asarray(genome, dtype=int).copy()
        for j in range(self.M):
            atribuidos = [
                i for i in range(self.N)
                if genome[i] > 0 and self._csm_de[i, genome[i]] == j
            ]
            horas = sum(self._horas[i, genome[i]] for i in atribuidos)
            orcamento = self.csms[j].orcamento_horas
            if horas <= orcamento + 1e-9:
                continue
            # remove pior densidade primeiro
            atribuidos.sort(
                key=lambda i: self._valor[i, genome[i]] / max(self._horas[i, genome[i]], 1e-9)
            )
            for i in atribuidos:
                if horas <= orcamento + 1e-9:
                    break
                horas -= self._horas[i, genome[i]]
                genome[i] = 0
        return genome

    def _linha_detalhe(self, i: int, gene: int) -> dict:
        c = self.customers[i]
        csm_idx, pb_idx = self.decode(gene)
        return {
            "cliente": c.nome,
            "mrr": c.mrr,
            "chs_atual": round(float(self._chs_atual[i]), 1),
            "classe_atual": classify_chs(float(self._chs_atual[i])),
            "acao": self.option_label(i, gene),
            "delta_chs": round(float(self._delta_chs[i, gene]), 1) if gene > 0 else 0.0,
            "horas": float(self._horas[i, gene]) if gene > 0 else 0.0,
            "valor": round(float(self._valor[i, gene]), 2) if gene > 0 else 0.0,
        }

    # ------------------------------------------------------------------------
    # UTILITÁRIOS PARA OS ALGORITMOS
    # ------------------------------------------------------------------------
    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.n_options, size=self.N)

    def chs_atual(self) -> np.ndarray:
        return self._chs_atual.copy()

    @property
    def capacidade_total(self) -> float:
        return sum(c.orcamento_horas for c in self.csms)

    @property
    def demanda_total_horas(self) -> float:
        """Horas necessárias se todo cliente recebesse o playbook mais barato."""
        custo_min = min(p.custo_horas for p in self.playbooks)
        return self.N * custo_min


# ============================================================================
# FÁBRICA: PLAYBOOKS, CSMs E PORTFÓLIO SINTÉTICO PADRÃO
# ============================================================================
def default_playbooks() -> List[Playbook]:
    """5 playbooks reais que exercitam diferentes variáveis de entrada."""
    return [
        Playbook("onb", "Onboarding Refresher", 4, {"engajamento": +15, "satisfacao": +1}),
        Playbook("ebr", "Executive Business Review", 8, {"satisfacao": +2, "engajamento": +5, "volume_suporte": -2}),
        Playbook("blitz", "Support Blitz", 6, {"volume_suporte": -10, "satisfacao": +1}),
        Playbook("adopt", "Adoption Coaching", 5, {"engajamento": +12, "volume_suporte": -3}),
        Playbook("bill", "Billing & Renewal Outreach", 3, {"saude_financeira": -20, "satisfacao": +1}),
    ]


def default_csms() -> List[CSM]:
    """Time de 5 CSMs com especialidades e orçamentos distintos.

    Capacidade total deliberadamente MENOR que a demanda: força o sistema a
    escolher QUEM atender e COMO rotear — o coração do problema de alocação.
    """
    return [
        CSM("csm1", "Ana (Onboarding)", 30, "Onboarding"),
        CSM("csm2", "Bruno (Retenção)", 24, "Retenção"),
        CSM("csm3", "Carla (Retenção)", 22, "Retenção"),
        CSM("csm4", "Diego (Adoção)", 28, "Adoção"),
        CSM("csm5", "Elena (Adoção)", 24, "Adoção"),
    ]


def default_specialty_mult() -> Dict[Tuple[str, str], float]:
    """Multiplicador de eficácia: especialidade do CSM × playbook.

    Acopla o trio (cliente, CSM, playbook) → torna o problema um GAP real.
    """
    tabela = {
        # especialidade:   onb   ebr   blitz  adopt  bill
        "Onboarding":     {"onb": 1.5, "ebr": 0.7, "blitz": 0.6, "adopt": 1.0, "bill": 0.8},
        "Retenção":       {"onb": 0.7, "ebr": 1.5, "blitz": 1.3, "adopt": 0.8, "bill": 1.2},
        "Adoção":         {"onb": 1.0, "ebr": 0.8, "blitz": 0.9, "adopt": 1.5, "bill": 0.7},
    }
    return {(esp, pid): f for esp, row in tabela.items() for pid, f in row.items()}


def make_default_problem(
    n_customers: int = 40,
    seed: int = 7,
    fitness_mode: str = "fuzzy",
    engine: Optional[HealthSenseEngine] = None,
) -> AllocationProblem:
    """Cria o problema padrão com portfólio sintético estruturado.

    O portfólio é desenhado de propósito com contas em pontos DIFERENTES da curva
    de CHS (crítica, em risco, fronteira, saudável). Assim a NÃO-LINEARIDADE do
    fuzzy realmente muda a alocação ótima: um +15 de engajamento rende muito mais
    ΔCHS numa conta de fronteira do que numa já saudável (saturação). Isso é o que
    sustenta a análise de impacto Fuzzy vs Linear (pontuação extra).
    """
    rng = np.random.default_rng(seed)
    customers: List[Customer] = []

    # Arquétipos: (eng, sup, sat, fin, ten) base + faixa de MRR + quantos
    arquetipos = [
        # nome,            eng, sup, sat, fin, ten,  mrr_lo, mrr_hi, frac
        ("Crítica",          8,  18,  3,  45,  20,    800,  4500, 0.15),
        ("Em risco",        30,  10,  5,  20,  16,   1500,  6000, 0.20),
        ("Fronteira",       50,   6,  7,   8,  14,   1000,  5000, 0.30),  # onde o fuzzy mais "morde"
        ("Saudável",        74,   3,  8,   2,  28,    600,  3500, 0.20),
        ("Promotora",       90,   1, 10,   0,  40,    500,  3000, 0.15),
    ]

    idx = 1
    for nome_arq, eng, sup, sat, fin, ten, mrr_lo, mrr_hi, frac in arquetipos:
        qtd = max(1, round(n_customers * frac))
        for _ in range(qtd):
            customers.append(Customer(
                cid=f"C{idx:03d}",
                nome=f"{nome_arq}-{idx:03d}",
                engajamento=float(np.clip(eng + rng.normal(0, 6), 0, 100)),
                volume_suporte=float(np.clip(sup + rng.normal(0, 3), 0, 30)),
                satisfacao=float(np.clip(sat + rng.normal(0, 1), 0, 10)),
                saude_financeira=float(np.clip(fin + rng.normal(0, 6), 0, 90)),
                tenure=float(np.clip(ten + rng.normal(0, 5), 0, 60)),
                mrr=float(round(rng.uniform(mrr_lo, mrr_hi), -1)),
            ))
            idx += 1

    return AllocationProblem(
        customers=customers,
        csms=default_csms(),
        playbooks=default_playbooks(),
        specialty_mult=default_specialty_mult(),
        engine=engine,
        fitness_mode=fitness_mode,
    )
