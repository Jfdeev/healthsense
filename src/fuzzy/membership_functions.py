"""
Funções de pertinência (Membership Functions) do sistema HealthSense.

Universos de discurso e termos linguísticos para 5 entradas e 1 saída.
Todos os parâmetros podem ser sobrescritos para análise de sensibilidade
e otimização via AG/PSO (ver notebooks/04_otimizacao_pso.ipynb).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ============================================================================
# UNIVERSOS DE DISCURSO
# ============================================================================
UNIVERSO_ENGAJAMENTO = np.arange(0, 101, 1)          # 0-100 (% índice de uso)
UNIVERSO_VOLUME_SUPORTE = np.arange(0, 31, 1)        # 0-30 (tickets/mês)
UNIVERSO_SATISFACAO = np.arange(0, 10.01, 0.1)       # 0-10 (NPS individual)
UNIVERSO_SAUDE_FINANCEIRA = np.arange(0, 91, 1)      # 0-90 (dias de atraso)
UNIVERSO_TENURE = np.arange(0, 61, 1)                # 0-60 (meses de contrato)
UNIVERSO_CHS = np.arange(0, 101, 1)                  # 0-100 (score final)


# ============================================================================
# PARÂMETROS DAS FUNÇÕES DE PERTINÊNCIA (default — passíveis de otimização)
# ============================================================================
@dataclass
class MFParams:
    """Parâmetros das funções de pertinência (otimizáveis via PSO/AG)."""

    # ENGAJAMENTO (trapezoidal nos extremos, triangular no meio)
    engajamento_baixo: tuple = (0, 0, 15, 35)
    engajamento_moderado: tuple = (25, 45, 65)
    engajamento_alto: tuple = (55, 70, 85)
    engajamento_muito_alto: tuple = (75, 90, 100, 100)

    # VOLUME DE SUPORTE
    suporte_saudavel: tuple = (0, 0, 2, 5)
    suporte_atencao: tuple = (3, 7, 12)
    suporte_alto: tuple = (10, 15, 22)
    suporte_critico: tuple = (18, 25, 30, 30)

    # SATISFAÇÃO (NPS)
    satisfacao_detrator: tuple = (0, 0, 4, 6)
    satisfacao_neutro: tuple = (5, 7, 9)
    satisfacao_promotor: tuple = (7.5, 9, 10, 10)

    # SAÚDE FINANCEIRA (dias de atraso de pagamento)
    financeira_em_dia: tuple = (0, 0, 5, 15)
    financeira_atraso_leve: tuple = (10, 25, 45)
    financeira_atraso_grave: tuple = (35, 60, 90, 90)

    # TENURE (meses de contrato)
    tenure_novo: tuple = (0, 0, 3, 9)
    tenure_estabelecido: tuple = (6, 18, 36)
    tenure_veterano: tuple = (30, 48, 60, 60)

    # CUSTOMER HEALTH SCORE — saída (gaussianas para transição suave)
    chs_critico: tuple = (10, 8)        # (mean, sigma)
    chs_em_risco: tuple = (30, 8)
    chs_atencao: tuple = (50, 8)
    chs_saudavel: tuple = (70, 8)
    chs_promotor: tuple = (90, 8)


# ============================================================================
# CONSTRUTORES DE VARIÁVEIS FUZZY
# ============================================================================
def build_engajamento(params: MFParams) -> ctrl.Antecedent:
    """Engajamento — índice composto: % dias ativos × adoção de features.

    Termos: {Baixo, Moderado, Alto, Muito Alto}
    Justificativa: 4 termos capturam transições suaves entre estado de
    cliente "morto" (≤15%), em ramp-up, saudável e power-user.
    """
    var = ctrl.Antecedent(UNIVERSO_ENGAJAMENTO, "engajamento")
    var["baixo"] = fuzz.trapmf(var.universe, list(params.engajamento_baixo))
    var["moderado"] = fuzz.trimf(var.universe, list(params.engajamento_moderado))
    var["alto"] = fuzz.trimf(var.universe, list(params.engajamento_alto))
    var["muito_alto"] = fuzz.trapmf(var.universe, list(params.engajamento_muito_alto))
    return var


def build_volume_suporte(params: MFParams) -> ctrl.Antecedent:
    """Volume de suporte — tickets abertos nos últimos 30 dias.

    Termos: {Saudável, Atenção, Alto, Crítico}
    Justificativa: muitos tickets é sintoma forte de fricção; 4 termos
    permitem distinguir "ruído normal" de "incêndio operacional".
    """
    var = ctrl.Antecedent(UNIVERSO_VOLUME_SUPORTE, "volume_suporte")
    var["saudavel"] = fuzz.trapmf(var.universe, list(params.suporte_saudavel))
    var["atencao"] = fuzz.trimf(var.universe, list(params.suporte_atencao))
    var["alto"] = fuzz.trimf(var.universe, list(params.suporte_alto))
    var["critico"] = fuzz.trapmf(var.universe, list(params.suporte_critico))
    return var


def build_satisfacao(params: MFParams) -> ctrl.Antecedent:
    """Satisfação — escala NPS individual de 0 a 10.

    Termos: {Detrator, Neutro, Promotor} — alinhado à definição NPS clássica
    (0-6 detrator, 7-8 neutro, 9-10 promotor), com transição suave fuzzy.
    """
    var = ctrl.Antecedent(UNIVERSO_SATISFACAO, "satisfacao")
    var["detrator"] = fuzz.trapmf(var.universe, list(params.satisfacao_detrator))
    var["neutro"] = fuzz.trimf(var.universe, list(params.satisfacao_neutro))
    var["promotor"] = fuzz.trapmf(var.universe, list(params.satisfacao_promotor))
    return var


def build_saude_financeira(params: MFParams) -> ctrl.Antecedent:
    """Saúde financeira — dias médios de atraso em pagamentos.

    Termos: {Em dia, Atraso leve, Atraso grave}
    Justificativa: inadimplência é preditor forte de churn; 3 termos
    bastam pois a relevância fica saturada após ~60 dias.
    """
    var = ctrl.Antecedent(UNIVERSO_SAUDE_FINANCEIRA, "saude_financeira")
    var["em_dia"] = fuzz.trapmf(var.universe, list(params.financeira_em_dia))
    var["atraso_leve"] = fuzz.trimf(var.universe, list(params.financeira_atraso_leve))
    var["atraso_grave"] = fuzz.trapmf(var.universe, list(params.financeira_atraso_grave))
    return var


def build_tenure(params: MFParams) -> ctrl.Antecedent:
    """Tenure — meses desde o início do contrato.

    Termos: {Novo, Estabelecido, Veterano}
    Justificativa: cliente novo tem comportamento diferente (onboarding);
    estabelecido (6-36 meses) é a fase "produtiva"; veterano (>30 meses)
    indica lealdade — perda dele é especialmente custosa.
    """
    var = ctrl.Antecedent(UNIVERSO_TENURE, "tenure")
    var["novo"] = fuzz.trapmf(var.universe, list(params.tenure_novo))
    var["estabelecido"] = fuzz.trimf(var.universe, list(params.tenure_estabelecido))
    var["veterano"] = fuzz.trapmf(var.universe, list(params.tenure_veterano))
    return var


def build_chs(params: MFParams) -> ctrl.Consequent:
    """Customer Health Score — saída final (0-100).

    Termos: {Crítico, Em Risco, Atenção, Saudável, Promotor}
    Justificativa: 5 termos com MFs gaussianas garantem saída contínua
    e suave; defuzzificação por centróide produz score numérico estável.
    """
    var = ctrl.Consequent(UNIVERSO_CHS, "customer_health_score", defuzzify_method="centroid")
    var["critico"] = fuzz.gaussmf(var.universe, *params.chs_critico)
    var["em_risco"] = fuzz.gaussmf(var.universe, *params.chs_em_risco)
    var["atencao"] = fuzz.gaussmf(var.universe, *params.chs_atencao)
    var["saudavel"] = fuzz.gaussmf(var.universe, *params.chs_saudavel)
    var["promotor"] = fuzz.gaussmf(var.universe, *params.chs_promotor)
    return var


# ============================================================================
# FACTORY DE TODAS AS VARIÁVEIS
# ============================================================================
def build_all_variables(params: MFParams | None = None) -> dict:
    """Constrói todas as variáveis fuzzy a partir de um conjunto de parâmetros."""
    if params is None:
        params = MFParams()
    return {
        "engajamento": build_engajamento(params),
        "volume_suporte": build_volume_suporte(params),
        "satisfacao": build_satisfacao(params),
        "saude_financeira": build_saude_financeira(params),
        "tenure": build_tenure(params),
        "chs": build_chs(params),
    }


def classify_chs(score: float) -> str:
    """Converte score numérico (0-100) em rótulo linguístico dominante.

    Útil para apresentação no dashboard e em testes.
    """
    if score < 20:
        return "Crítico"
    if score < 40:
        return "Em Risco"
    if score < 60:
        return "Atenção"
    if score < 80:
        return "Saudável"
    return "Promotor"
