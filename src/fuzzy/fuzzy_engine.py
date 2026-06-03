"""
Engine fuzzy Mamdani do HealthSense.

Encapsula o sistema de controle scikit-fuzzy, expondo uma API simples:
    engine = HealthSenseEngine()
    resultado = engine.evaluate(engajamento=80, volume_suporte=2, satisfacao=9,
                                 saude_financeira=0, tenure=18)
    # → {"chs": 87.3, "classificacao": "Promotor", "regras_ativadas": [...]}

Inferência: Mamdani
  - AND  → min (T-norma de Mamdani)
  - OR   → max (S-norma)
  - Implicação → min
  - Agregação → max
  - Defuzzificação → centróide
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from .membership_functions import MFParams, build_all_variables, classify_chs
from .rules import build_rules, RuleSpec


@dataclass
class EvaluationResult:
    inputs: dict
    chs: float
    classificacao: str
    activations: dict  # rule_id -> grau de ativação (0-1)


class HealthSenseEngine:
    """Sistema fuzzy Mamdani completo para CHS."""

    def __init__(self, params: Optional[MFParams] = None):
        self.params = params or MFParams()
        self.variables = build_all_variables(self.params)
        self.rule_specs = build_rules(self.variables)
        self._control_system = ctrl.ControlSystem([r.rule for r in self.rule_specs])

    # ------------------------------------------------------------------------
    # API PRINCIPAL
    # ------------------------------------------------------------------------
    def evaluate(
        self,
        engajamento: float,
        volume_suporte: float,
        satisfacao: float,
        saude_financeira: float,
        tenure: float,
    ) -> EvaluationResult:
        """Avalia uma única amostra. Retorna score, classificação e ativações."""
        sim = ctrl.ControlSystemSimulation(self._control_system)
        sim.input["engajamento"] = self._clip(engajamento, 0, 100)
        sim.input["volume_suporte"] = self._clip(volume_suporte, 0, 30)
        sim.input["satisfacao"] = self._clip(satisfacao, 0, 10)
        sim.input["saude_financeira"] = self._clip(saude_financeira, 0, 90)
        sim.input["tenure"] = self._clip(tenure, 0, 60)
        sim.compute()

        # Defesa: scikit-fuzzy lança KeyError se NENHUMA regra ativar.
        # Com as regras baseline (R29-R32) isso não deveria acontecer,
        # mas mantemos fallback neutro (Atenção) para robustez.
        try:
            chs_value = float(sim.output["customer_health_score"])
        except KeyError:
            chs_value = 50.0

        return EvaluationResult(
            inputs={
                "engajamento": engajamento,
                "volume_suporte": volume_suporte,
                "satisfacao": satisfacao,
                "saude_financeira": saude_financeira,
                "tenure": tenure,
            },
            chs=chs_value,
            classificacao=classify_chs(chs_value),
            activations=self._compute_activations(
                engajamento, volume_suporte, satisfacao, saude_financeira, tenure
            ),
        )

    def evaluate_batch(self, df) -> list[EvaluationResult]:
        """Avalia um DataFrame com colunas: engajamento, volume_suporte, satisfacao,
        saude_financeira, tenure."""
        results = []
        for _, row in df.iterrows():
            results.append(self.evaluate(
                engajamento=row["engajamento"],
                volume_suporte=row["volume_suporte"],
                satisfacao=row["satisfacao"],
                saude_financeira=row["saude_financeira"],
                tenure=row["tenure"],
            ))
        return results

    # ------------------------------------------------------------------------
    # ANÁLISE
    # ------------------------------------------------------------------------
    def control_surface(
        self,
        x_var: str,
        y_var: str,
        fixed: dict,
        resolution: int = 25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gera superfície de controle 2D varrendo x_var × y_var, fixando as demais."""
        ranges = {
            "engajamento": (0, 100),
            "volume_suporte": (0, 30),
            "satisfacao": (0, 10),
            "saude_financeira": (0, 90),
            "tenure": (0, 60),
        }
        x_range = np.linspace(*ranges[x_var], resolution)
        y_range = np.linspace(*ranges[y_var], resolution)
        Z = np.zeros((resolution, resolution))

        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                kwargs = dict(fixed)
                kwargs[x_var] = float(x)
                kwargs[y_var] = float(y)
                result = self.evaluate(**kwargs)
                Z[j, i] = result.chs  # j=linhas (y), i=colunas (x)

        X, Y = np.meshgrid(x_range, y_range)
        return X, Y, Z

    # ------------------------------------------------------------------------
    # INTERNO
    # ------------------------------------------------------------------------
    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    def _compute_activations(
        self,
        engajamento: float,
        volume_suporte: float,
        satisfacao: float,
        saude_financeira: float,
        tenure: float,
    ) -> dict:
        """Calcula o grau de ativação de cada regra (útil para explicabilidade)."""
        memberships = {
            "engajamento": self._membership_for("engajamento", engajamento),
            "volume_suporte": self._membership_for("volume_suporte", volume_suporte),
            "satisfacao": self._membership_for("satisfacao", satisfacao),
            "saude_financeira": self._membership_for("saude_financeira", saude_financeira),
            "tenure": self._membership_for("tenure", tenure),
        }

        # Mapeamento manual das regras para cálculo de ativação
        # (espelho do que está em rules.py)
        rule_activation_specs = {
            "R1": [("engajamento", "baixo"), ("volume_suporte", "critico")],
            "R2": [("satisfacao", "detrator"), ("saude_financeira", "atraso_grave")],
            "R3": [("engajamento", "baixo"), ("saude_financeira", "atraso_grave")],
            "R4": [("tenure", "veterano"), ("engajamento", "baixo"), ("satisfacao", "detrator")],
            "R5": [("engajamento", "baixo"), ("tenure", "veterano")],
            "R6": [("satisfacao", "detrator"), ("volume_suporte", "alto")],
            "R7": [("volume_suporte", "critico"), ("saude_financeira", "atraso_leve")],
            "R8": [("engajamento", "moderado"), ("satisfacao", "detrator")],
            "R9": [("engajamento", "baixo"), ("tenure", "estabelecido")],
            "R10": [("tenure", "veterano"), ("satisfacao", "detrator")],
            "R11": [("tenure", "novo"), ("engajamento", "moderado")],
            "R12": [("engajamento", "moderado"), ("volume_suporte", "atencao")],
            "R13": [("saude_financeira", "atraso_leve"), ("satisfacao", "neutro")],
            "R14": [("tenure", "novo"), ("volume_suporte", "alto")],
            "R15": [("engajamento", "alto"), ("satisfacao", "detrator")],
            "R16": [("engajamento", "baixo"), ("tenure", "novo")],
            "R17": [("engajamento", "alto"), ("satisfacao", "neutro"), ("saude_financeira", "em_dia")],
            "R18": [("engajamento", "alto"), ("volume_suporte", "saudavel")],
            "R19": [("tenure", "veterano"), ("engajamento", "moderado"), ("saude_financeira", "em_dia")],
            "R20": [("engajamento", "alto"), ("satisfacao", "promotor")],
            "R21": [("engajamento", "muito_alto"), ("volume_suporte", "atencao")],
            "R22": [("engajamento", "muito_alto"), ("satisfacao", "promotor"), ("saude_financeira", "em_dia")],
            "R23": [("engajamento", "muito_alto"), ("satisfacao", "promotor"), ("tenure", "veterano")],
            "R24": [("engajamento", "muito_alto"), ("volume_suporte", "saudavel"), ("saude_financeira", "em_dia")],
            "R25": [("engajamento", "alto"), ("satisfacao", "promotor"), ("tenure", "veterano"), ("saude_financeira", "em_dia")],
            "R26": [("engajamento", "muito_alto"), ("saude_financeira", "atraso_grave")],
            "R27": [("satisfacao", "promotor"), ("volume_suporte", "critico")],
            "R28": [("engajamento", "moderado"), ("satisfacao", "promotor"), ("volume_suporte", "saudavel")],
            "R29": [("engajamento", "baixo")],
            "R30": [("engajamento", "moderado")],
            "R31": [("engajamento", "alto")],
            "R32": [("engajamento", "muito_alto")],
        }

        activations = {}
        for rule_id, terms in rule_activation_specs.items():
            # AND = min entre as pertinências
            grau = min(memberships[var][term] for var, term in terms)
            activations[rule_id] = round(float(grau), 4)
        return activations

    def _membership_for(self, var_name: str, value: float) -> dict:
        """Calcula a pertinência de `value` em cada termo linguístico de `var_name`."""
        var = self.variables[var_name]
        value = self._clip(
            value,
            float(var.universe.min()),
            float(var.universe.max()),
        )
        result = {}
        for term_name, term_obj in var.terms.items():
            result[term_name] = float(
                fuzz.interp_membership(var.universe, term_obj.mf, value)
            )
        return result
