"""
Motor fuzzy Mamdani — REUSADO (vendorado) da Parte 1 do trabalho.

Origem: HealthSense Customer Health Score (Sistema de Controle Fuzzy).
Aqui ele é usado como FUNÇÃO DE FITNESS do otimizador evolutivo — atende a
pontuação extra "Alternativa 2: Integração Fuzzy-Evolutiva".

Não modificar a lógica fuzzy aqui: este código é idêntico ao da Parte 1 para
manter coerência entre os dois trabalhos. Qualquer ajuste deve ser feito na
Parte 1 e re-copiado.
"""

from .fuzzy_engine import HealthSenseEngine, EvaluationResult
from .membership_functions import MFParams, classify_chs

__all__ = ["HealthSenseEngine", "EvaluationResult", "MFParams", "classify_chs"]
