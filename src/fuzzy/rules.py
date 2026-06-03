"""
Base de regras Mamdani do HealthSense — 28 regras com justificativa de domínio.

Cada regra tem:
  - ID (R1, R2, ...)
  - Categoria (CRITICO, RISCO, ATENCAO, SAUDAVEL, PROMOTOR, CONFLITO)
  - Justificativa textual baseada em práticas de Customer Success B2B

Atende a trilha de ampliação obrigatória de grupos de 5 (mínimo 18 regras).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from skfuzzy import control as ctrl


@dataclass
class RuleSpec:
    rule_id: str
    categoria: str
    justificativa: str
    rule: ctrl.Rule


def build_rules(vars: dict) -> List[RuleSpec]:
    """Constrói a base de 28 regras a partir das variáveis fuzzy."""
    eng = vars["engajamento"]
    sup = vars["volume_suporte"]
    sat = vars["satisfacao"]
    fin = vars["saude_financeira"]
    ten = vars["tenure"]
    chs = vars["chs"]

    rules: List[RuleSpec] = []

    # ------------------------------------------------------------------------
    # CATEGORIA: CRÍTICO (cliente prestes a cancelar)
    # ------------------------------------------------------------------------
    rules.append(RuleSpec(
        "R1", "CRITICO",
        "Engajamento baixo + tickets críticos = cliente abandonado em meio a problema operacional grave.",
        ctrl.Rule(eng["baixo"] & sup["critico"], chs["critico"]),
    ))
    rules.append(RuleSpec(
        "R2", "CRITICO",
        "Detrator (NPS baixo) com inadimplência grave = vontade explícita de sair + sinal financeiro.",
        ctrl.Rule(sat["detrator"] & fin["atraso_grave"], chs["critico"]),
    ))
    rules.append(RuleSpec(
        "R3", "CRITICO",
        "Engajamento baixo + atraso grave = cliente fora da plataforma E inadimplente.",
        ctrl.Rule(eng["baixo"] & fin["atraso_grave"], chs["critico"]),
    ))
    rules.append(RuleSpec(
        "R4", "CRITICO",
        "Veterano com baixo engajamento, suporte crítico e detrator = churn iminente de conta estratégica.",
        ctrl.Rule(ten["veterano"] & eng["baixo"] & sat["detrator"], chs["critico"]),
    ))

    # ------------------------------------------------------------------------
    # CATEGORIA: EM RISCO (sinal forte de churn, intervenção urgente)
    # ------------------------------------------------------------------------
    rules.append(RuleSpec(
        "R5", "RISCO",
        "Veterano com engajamento baixo = lealdade artificial, alto risco de não-renovação.",
        ctrl.Rule(eng["baixo"] & ten["veterano"], chs["em_risco"]),
    ))
    rules.append(RuleSpec(
        "R6", "RISCO",
        "Detrator com alto volume de suporte = frustrado com fricções operacionais.",
        ctrl.Rule(sat["detrator"] & sup["alto"], chs["em_risco"]),
    ))
    rules.append(RuleSpec(
        "R7", "RISCO",
        "Suporte crítico + atraso leve = dor operacional contagiando saúde financeira.",
        ctrl.Rule(sup["critico"] & fin["atraso_leve"], chs["em_risco"]),
    ))
    rules.append(RuleSpec(
        "R8", "RISCO",
        "Engajamento moderado + detrator = uso mecânico sem valor percebido.",
        ctrl.Rule(eng["moderado"] & sat["detrator"], chs["em_risco"]),
    ))
    rules.append(RuleSpec(
        "R9", "RISCO",
        "Engajamento baixo em cliente estabelecido = sinal de adoção que estagnou.",
        ctrl.Rule(eng["baixo"] & ten["estabelecido"], chs["em_risco"]),
    ))
    rules.append(RuleSpec(
        "R10", "RISCO",
        "Veterano detrator = relacionamento azedando após longo histórico.",
        ctrl.Rule(ten["veterano"] & sat["detrator"], chs["em_risco"]),
    ))

    # ------------------------------------------------------------------------
    # CATEGORIA: ATENÇÃO (cliente em zona neutra/incerta, monitoramento ativo)
    # ------------------------------------------------------------------------
    rules.append(RuleSpec(
        "R11", "ATENCAO",
        "Cliente novo com engajamento moderado = onboarding em progresso, ainda imaturo.",
        ctrl.Rule(ten["novo"] & eng["moderado"], chs["atencao"]),
    ))
    rules.append(RuleSpec(
        "R12", "ATENCAO",
        "Engajamento moderado + suporte em atenção = adoção parcial com fricção média.",
        ctrl.Rule(eng["moderado"] & sup["atencao"], chs["atencao"]),
    ))
    rules.append(RuleSpec(
        "R13", "ATENCAO",
        "Atraso leve + satisfação neutra = cliente sem reclamação mas com sinais financeiros amarelos.",
        ctrl.Rule(fin["atraso_leve"] & sat["neutro"], chs["atencao"]),
    ))
    rules.append(RuleSpec(
        "R14", "ATENCAO",
        "Cliente novo recebendo muito suporte = onboarding com dificuldade técnica.",
        ctrl.Rule(ten["novo"] & sup["alto"], chs["atencao"]),
    ))
    rules.append(RuleSpec(
        "R15", "ATENCAO",
        "Engajamento alto mas detrator = usa o produto contrariado (alto risco de migração).",
        ctrl.Rule(eng["alto"] & sat["detrator"], chs["atencao"]),
    ))
    rules.append(RuleSpec(
        "R16", "ATENCAO",
        "Cliente novo com baixo engajamento = onboarding falhando, intervenção early-stage.",
        ctrl.Rule(eng["baixo"] & ten["novo"], chs["atencao"]),
    ))

    # ------------------------------------------------------------------------
    # CATEGORIA: SAUDÁVEL (cliente em estado normal, manter)
    # ------------------------------------------------------------------------
    rules.append(RuleSpec(
        "R17", "SAUDAVEL",
        "Engajamento alto, satisfação neutra e financeiro em dia = padrão de cliente estável.",
        ctrl.Rule(eng["alto"] & sat["neutro"] & fin["em_dia"], chs["saudavel"]),
    ))
    rules.append(RuleSpec(
        "R18", "SAUDAVEL",
        "Engajamento alto + baixo volume de suporte = adoção saudável e auto-suficiente.",
        ctrl.Rule(eng["alto"] & sup["saudavel"], chs["saudavel"]),
    ))
    rules.append(RuleSpec(
        "R19", "SAUDAVEL",
        "Veterano com engajamento moderado e em dia = padrão maduro, uso eficiente.",
        ctrl.Rule(ten["veterano"] & eng["moderado"] & fin["em_dia"], chs["saudavel"]),
    ))
    rules.append(RuleSpec(
        "R20", "SAUDAVEL",
        "Engajamento alto + promotor = cliente satisfeito usando ativamente.",
        ctrl.Rule(eng["alto"] & sat["promotor"], chs["saudavel"]),
    ))
    rules.append(RuleSpec(
        "R21", "SAUDAVEL",
        "Engajamento muito alto com leve fricção de suporte = power user com ruído normal.",
        ctrl.Rule(eng["muito_alto"] & sup["atencao"], chs["saudavel"]),
    ))

    # ------------------------------------------------------------------------
    # CATEGORIA: PROMOTOR (cliente embaixador, candidato a expansão/upsell)
    # ------------------------------------------------------------------------
    rules.append(RuleSpec(
        "R22", "PROMOTOR",
        "Engajamento muito alto + promotor + em dia = perfil ideal para expansão de receita.",
        ctrl.Rule(eng["muito_alto"] & sat["promotor"] & fin["em_dia"], chs["promotor"]),
    ))
    rules.append(RuleSpec(
        "R23", "PROMOTOR",
        "Engajamento muito alto + promotor + veterano = case de sucesso, candidato a referência.",
        ctrl.Rule(eng["muito_alto"] & sat["promotor"] & ten["veterano"], chs["promotor"]),
    ))
    rules.append(RuleSpec(
        "R24", "PROMOTOR",
        "Power user sem suporte e em dia = adoção orgânica, alta autonomia.",
        ctrl.Rule(eng["muito_alto"] & sup["saudavel"] & fin["em_dia"], chs["promotor"]),
    ))
    rules.append(RuleSpec(
        "R25", "PROMOTOR",
        "Veterano promotor com engajamento alto e em dia = relacionamento maduro de valor.",
        ctrl.Rule(eng["alto"] & sat["promotor"] & ten["veterano"] & fin["em_dia"], chs["promotor"]),
    ))

    # ------------------------------------------------------------------------
    # CATEGORIA: CONFLITO (regras de exceção/fronteira que evitam saídas absurdas)
    # ------------------------------------------------------------------------
    rules.append(RuleSpec(
        "R26", "CONFLITO",
        "Power user inadimplente grave = uso ativo não compensa risco financeiro, rebaixa para risco.",
        ctrl.Rule(eng["muito_alto"] & fin["atraso_grave"], chs["em_risco"]),
    ))
    rules.append(RuleSpec(
        "R27", "CONFLITO",
        "Promotor com suporte crítico = elogio ao produto mas dor operacional aguda, rebaixa para atenção.",
        ctrl.Rule(sat["promotor"] & sup["critico"], chs["atencao"]),
    ))
    rules.append(RuleSpec(
        "R28", "CONFLITO",
        "Engajamento moderado + promotor + suporte saudável = cliente satisfeito mesmo com uso parcial.",
        ctrl.Rule(eng["moderado"] & sat["promotor"] & sup["saudavel"], chs["saudavel"]),
    ))

    # ------------------------------------------------------------------------
    # CATEGORIA: BASELINE (regras de cobertura — garantem que TODA entrada
    # ative pelo menos uma regra; justificadas como heurística geral de
    # Customer Success quando nenhum padrão específico se aplica)
    # ------------------------------------------------------------------------
    rules.append(RuleSpec(
        "R29", "BASELINE",
        "Sem outros sinais, engajamento baixo é heurística de risco (uso é proxy primário de valor).",
        ctrl.Rule(eng["baixo"], chs["em_risco"]),
    ))
    rules.append(RuleSpec(
        "R30", "BASELINE",
        "Sem outros sinais, engajamento moderado é zona de atenção (adoção parcial).",
        ctrl.Rule(eng["moderado"], chs["atencao"]),
    ))
    rules.append(RuleSpec(
        "R31", "BASELINE",
        "Sem outros sinais, engajamento alto é heurística de saúde (uso ativo do produto).",
        ctrl.Rule(eng["alto"], chs["saudavel"]),
    ))
    rules.append(RuleSpec(
        "R32", "BASELINE",
        "Sem outros sinais, engajamento muito alto é heurística de saudável "
        "(promoção a 'Promotor' exige sinal explícito de NPS+financeiro nas regras específicas R22-R25).",
        ctrl.Rule(eng["muito_alto"], chs["saudavel"]),
    ))

    return rules


def get_rules_table(rules: List[RuleSpec]) -> List[dict]:
    """Retorna tabela tabulável da base de regras (para README e relatório)."""
    return [
        {
            "ID": r.rule_id,
            "Categoria": r.categoria,
            "Justificativa": r.justificativa,
        }
        for r in rules
    ]
