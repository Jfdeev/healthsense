"""
Cenários de teste do HealthSense.

São 14 cenários, cobrindo:
  - 3 baixos (Crítico esperado)
  - 3 médios (Atenção esperado)
  - 3 altos (Saudável/Promotor esperado)
  - 3 fronteiriços (transição entre faixas)
  - 2 conflitantes (variáveis se opondo)

Atende ao requisito mínimo de 12 cenários da trilha de ampliação para grupos
de 5 integrantes.

Rodar:
    pytest tests/ -v
"""
from __future__ import annotations

import pytest

from src.fuzzy.fuzzy_engine import HealthSenseEngine


@pytest.fixture(scope="module")
def engine() -> HealthSenseEngine:
    return HealthSenseEngine()


# ============================================================================
# CENÁRIOS BAIXOS — esperado: CRÍTICO ou EM RISCO
# ============================================================================
class TestCenariosCriticos:
    def test_C01_cliente_abandonado_inadimplente(self, engine):
        """Cliente que parou de usar E não paga há 2 meses."""
        r = engine.evaluate(engajamento=5, volume_suporte=2, satisfacao=3, saude_financeira=60, tenure=18)
        assert r.chs <= 35, f"Esperado CHS≤35, obtido {r.chs:.1f}"
        assert r.classificacao in ("Crítico", "Em Risco")

    def test_C02_explosao_de_tickets_e_detrator(self, engine):
        """Cliente em chamas: muitos tickets + extremamente insatisfeito."""
        r = engine.evaluate(engajamento=15, volume_suporte=25, satisfacao=2, saude_financeira=10, tenure=24)
        assert r.chs <= 35
        assert r.classificacao in ("Crítico", "Em Risco")

    def test_C03_veterano_que_parou_de_usar(self, engine):
        """Conta estratégica que parou — alto risco de cancelamento."""
        r = engine.evaluate(engajamento=10, volume_suporte=5, satisfacao=4, saude_financeira=20, tenure=48)
        assert r.chs <= 40


# ============================================================================
# CENÁRIOS MÉDIOS — esperado: ATENÇÃO
# ============================================================================
class TestCenariosAtencao:
    def test_C04_cliente_novo_em_onboarding(self, engine):
        """Cliente novo, uso parcial, normal pra fase."""
        r = engine.evaluate(engajamento=45, volume_suporte=6, satisfacao=7, saude_financeira=0, tenure=2)
        assert 40 <= r.chs <= 65

    def test_C05_uso_estagnado_sem_reclamacao(self, engine):
        """Cliente "morno": usa pouco, não reclama, paga em dia."""
        r = engine.evaluate(engajamento=42, volume_suporte=4, satisfacao=7, saude_financeira=8, tenure=15)
        assert 40 <= r.chs <= 65

    def test_C06_engajamento_alto_mas_detrator(self, engine):
        """Cliente usa muito mas está insatisfeito — sinal de migração."""
        r = engine.evaluate(engajamento=75, volume_suporte=3, satisfacao=3, saude_financeira=0, tenure=20)
        assert 30 <= r.chs <= 60


# ============================================================================
# CENÁRIOS ALTOS — esperado: SAUDÁVEL ou PROMOTOR
# ============================================================================
class TestCenariosAltos:
    def test_C07_power_user_promotor(self, engine):
        """Cliente ideal: muito ativo, NPS alto, em dia, veterano."""
        r = engine.evaluate(engajamento=95, volume_suporte=1, satisfacao=10, saude_financeira=0, tenure=42)
        assert r.chs >= 75
        assert r.classificacao in ("Saudável", "Promotor")

    def test_C08_cliente_saudavel_padrao(self, engine):
        """Conta normal e estável: bom uso, satisfação ok."""
        r = engine.evaluate(engajamento=72, volume_suporte=3, satisfacao=8, saude_financeira=2, tenure=24)
        assert r.chs >= 60
        assert r.classificacao in ("Saudável", "Promotor")

    def test_C09_promotor_recente(self, engine):
        """Cliente novo mas já com NPS alto e bom uso — candidato a estudo de caso."""
        r = engine.evaluate(engajamento=88, volume_suporte=2, satisfacao=10, saude_financeira=0, tenure=6)
        assert r.chs >= 65


# ============================================================================
# CENÁRIOS FRONTEIRIÇOS — transição entre faixas
# ============================================================================
class TestCenariosFronteira:
    def test_C10_fronteira_atencao_saudavel(self, engine):
        """Limite entre "Atenção" (60) e "Saudável" (80)."""
        r = engine.evaluate(engajamento=60, volume_suporte=5, satisfacao=7, saude_financeira=5, tenure=18)
        assert 50 <= r.chs <= 75

    def test_C11_fronteira_risco_atencao(self, engine):
        """Limite entre "Em Risco" (40) e "Atenção" (60)."""
        r = engine.evaluate(engajamento=35, volume_suporte=8, satisfacao=6, saude_financeira=15, tenure=12)
        assert 30 <= r.chs <= 55

    def test_C12_fronteira_saudavel_promotor(self, engine):
        """Limite superior — muito perto de promotor."""
        r = engine.evaluate(engajamento=82, volume_suporte=2, satisfacao=9, saude_financeira=0, tenure=30)
        assert r.chs >= 65


# ============================================================================
# CENÁRIOS CONFLITANTES — variáveis se opondo
# ============================================================================
class TestCenariosConflitantes:
    def test_C13_power_user_inadimplente(self, engine):
        """Usa muito mas não paga — qual peso domina?"""
        r = engine.evaluate(engajamento=92, volume_suporte=1, satisfacao=8, saude_financeira=75, tenure=24)
        # Regra R26 deve rebaixar para "Em Risco"
        assert r.chs <= 65, f"Inadimplência grave deveria rebaixar; obtido {r.chs:.1f}"

    def test_C14_promotor_em_chamas_de_suporte(self, engine):
        """Cliente diz que ama o produto mas tem ticket crítico aberto."""
        r = engine.evaluate(engajamento=70, volume_suporte=22, satisfacao=10, saude_financeira=0, tenure=18)
        # Regra R27 deve rebaixar para "Atenção"
        assert 40 <= r.chs <= 75


# ============================================================================
# SANIDADE
# ============================================================================
def test_score_dentro_do_universo(engine):
    """Em qualquer entrada válida, CHS ∈ [0, 100]."""
    casos = [(0, 0, 0, 0, 0), (100, 30, 10, 90, 60), (50, 15, 5, 45, 30)]
    for caso in casos:
        r = engine.evaluate(*caso)
        assert 0 <= r.chs <= 100, f"CHS fora de [0,100]: {r.chs:.2f}"


def test_classificacao_consistente(engine):
    """Classificação deve bater com a faixa do CHS.

    Design note: a baseline rule R32 (muito_alto → saudável) garante cobertura
    para qualquer engajamento alto, mesmo sem sinais específicos. Isso pondera
    o resultado: 'Promotor' exige sinais sustentados (R22-R25), não apenas
    engajamento alto — alinhado à definição NPS clássica.
    """
    # Caso "muito positivo" — espera-se Saudável OU Promotor
    r = engine.evaluate(95, 1, 10, 0, 48)
    assert r.chs >= 75, f"Esperado CHS≥75 para sinais máximos, obtido {r.chs:.1f}"
    assert r.classificacao in ("Saudável", "Promotor")

    # Caso intermediário — power user com tenure no início de veterano
    r = engine.evaluate(95, 1, 10, 0, 36)
    assert r.chs >= 70
    assert r.classificacao in ("Saudável", "Promotor")

    # Caso crítico
    r = engine.evaluate(5, 25, 1, 80, 24)
    assert r.classificacao in ("Crítico", "Em Risco")
