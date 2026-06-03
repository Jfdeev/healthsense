# HealthSense — Customer Health Score Fuzzy para SaaS B2B

**Trabalho de Pesquisa e Desenvolvimento — Parte 1: Sistemas de Controle Fuzzy**
**Disciplina:** Inteligência Artificial e Computacional (0700M8) — CESUPA 2026/1
**Professor:** Daniel Leal Souza
**Modalidade:** B (Aplicação/Produto) · **Modelo:** Mamdani · **Grupo de 5** com trilha de ampliação técnica
**Pontuação extra:** Otimização PSO de funções de pertinência

**Repositório GitHub:** _<inserir link>_

**Equipe:**
- _<Nome 1>_ — Product Owner / Pesquisa
- _<Nome 2>_ — Engenheiro Fuzzy
- _<Nome 3>_ — Dev Backend
- _<Nome 4>_ — Dev Frontend
- _<Nome 5>_ — QA + Otimização

---

## Resumo

Este trabalho apresenta o **HealthSense**, um sistema de apoio à decisão para Customer Success em SaaS B2B baseado em controle fuzzy Mamdani. O sistema combina 5 variáveis linguísticas (engajamento, volume de suporte, satisfação NPS, saúde financeira e tenure) através de 28 regras derivadas de práticas de mercado, produzindo um Customer Health Score (CHS) de 0 a 100 classificado em 5 níveis (Crítico, Em Risco, Atenção, Saudável, Promotor). Uma extensão opcional aplica PSO para ajuste automático dos parâmetros das funções de pertinência. O produto é entregue como dashboard Streamlit interativo, com explicabilidade (regras ativadas), superfícies de controle 3D e suíte de 14 cenários de teste.

**Palavras-chave:** lógica fuzzy, Mamdani, Customer Success, SaaS B2B, PSO.

---

## 1. Introdução e motivação

### 1.1. Contexto

O modelo de negócio SaaS B2B vive de **retenção e expansão de receita recorrente**. Equipes de Customer Success (CS) gerenciam carteiras de centenas de contas e precisam, diariamente, decidir quem priorizar: quem está em risco de churn? Quem está pronto para upsell? Quem pode virar caso de sucesso?

### 1.2. Problema

Soluções tradicionais para essa priorização sofrem de:

- **Scores aditivos com pesos arbitrários** — não capturam interações entre variáveis (cliente com alto NPS mas baixo uso não é "metade saudável").
- **Regras `if/else` rígidas** — quebram em casos de fronteira (atraso de 30 vs. 31 dias produzindo classificações opostas).
- **Modelos black-box (ML)** — difíceis de explicar a stakeholders não-técnicos.

### 1.3. Por que fuzzy?

A lógica fuzzy é adequada porque:

1. **Transições suaves**: clientes em estados intermediários recebem scores intermediários.
2. **Regras linguísticas interpretáveis**: aderem ao raciocínio do CSM ("se engajamento é baixo E suporte é crítico, ENTÃO o cliente está em risco").
3. **Combinação não-linear de variáveis** via base de regras: capta interações sem precisar modelar matematicamente cada uma.
4. **Explicabilidade nativa**: cada saída pode ser justificada pelas regras ativadas.

### 1.4. Objetivos

- **Geral:** implementar um sistema fuzzy Mamdani funcional, validado e demonstrável para CHS.
- **Específicos:**
  - Modelar 5 variáveis linguísticas e 1 saída com universos justificados.
  - Construir base de 28 regras com justificativa de domínio.
  - Entregar dashboard interativo como produto.
  - Validar com 14 cenários cobrindo casos típicos e críticos.
  - **(Extra)** Aplicar PSO para otimização automática de parâmetros.

---

## 2. Fundamentação teórica

### 2.1. Lógica fuzzy e conjuntos fuzzy

(Resumo conceitual baseado em Zadeh 1965, Klir & Yuan 1995. Citar fontes.)

### 2.2. Sistemas fuzzy Mamdani

Componentes: fuzzificação → base de regras → inferência → defuzzificação. Operadores típicos: AND=min, OR=max, implicação=min, agregação=max, defuzzificação=centróide.

### 2.3. Customer Health Score em SaaS

Discussão breve sobre o estado da prática (referenciar artigos/blogs de Gainsight, ChurnZero, HubSpot). Comparar com abordagens fuzzy publicadas para CHS (buscar 2-3 artigos JCR/Qualis em Web of Science ou Google Scholar).

### 2.4. Particle Swarm Optimization (PSO) — extra

(Resumo conceitual para a extensão de pontuação extra.)

---

## 3. Análise de mercado e requisitos

### 3.1. Público-alvo

- Customer Success Managers (CSMs)
- Heads of Customer Success
- Times de Renovação / Expansion

### 3.2. Requisitos funcionais

| ID | Requisito |
|---|---|
| RF01 | Calcular CHS a partir de 5 variáveis de entrada |
| RF02 | Classificar cliente em 5 níveis |
| RF03 | Exibir gauge visual com cor por faixa |
| RF04 | Mostrar regras ativadas (explicabilidade) |
| RF05 | Gerar superfície de controle 2D de quaisquer duas variáveis |
| RF06 | Recomendar ação textual por categoria |

### 3.3. Requisitos não-funcionais

| ID | Requisito |
|---|---|
| RNF01 | Resposta em < 200 ms por avaliação |
| RNF02 | Interface acessível via navegador (Streamlit) |
| RNF03 | Reproduzível em qualquer máquina com Python 3.10+ |
| RNF04 | Código aberto e versionado em GitHub |

### 3.4. Limitações e riscos de interpretação incorreta

- CHS é apoio à decisão, não diagnóstico determinístico.
- Pesos das regras refletem práticas de mercado, podem variar entre empresas.
- Dados sintéticos foram usados em validação (não validamos com cliente real).

---

## 4. Modelagem fuzzy

### 4.1. Variáveis de entrada

(Tabela detalhada — copiar do `README.md` + justificativa de cada universo e termo)

### 4.2. Variável de saída

| Termo | Faixa típica do CHS | Significado de negócio |
|---|---|---|
| Crítico | 0–20 | Cancelamento iminente |
| Em Risco | 20–40 | Intervenção urgente |
| Atenção | 40–60 | Monitoramento ativo |
| Saudável | 60–80 | Estado normal |
| Promotor | 80–100 | Candidato a expansão/referência |

### 4.3. Funções de pertinência

Inserir aqui:
- Tabela com parâmetros de cada MF (de `MFParams` em `src/membership_functions.py`)
- Gráficos das MFs (gerar via `notebooks/01_modelagem.ipynb` e salvar em `assets/mf_plots/`)
- Justificativa por escolha (trapezoidal nos extremos, triangular nos intermediários, gaussiana na saída)

### 4.4. Base de regras

Tabela completa das 28 regras (copiar de `src/rules.py`) agrupadas por categoria.

### 4.5. Mecanismo de inferência

- **AND**: min (T-norma de Mamdani)
- **OR**: max
- **Implicação**: min
- **Agregação**: max
- **Defuzzificação**: centróide (justificativa: produz saída numérica contínua e suave, mais adequada que MOM ou bisector para um score)

---

## 5. Implementação

### 5.1. Arquitetura do sistema

(Diagrama: usuário → Streamlit → HealthSenseEngine → scikit-fuzzy → resposta)

### 5.2. Stack tecnológico

(Copiar de README.md)

### 5.3. Estrutura do código

(Copiar tree do repositório)

### 5.4. API principal

```python
from src.fuzzy_engine import HealthSenseEngine

engine = HealthSenseEngine()
resultado = engine.evaluate(
    engajamento=80, volume_suporte=2, satisfacao=9,
    saude_financeira=0, tenure=18
)
# EvaluationResult(chs=87.3, classificacao='Promotor', activations={...})
```

### 5.5. Dashboard

(Screenshots do Streamlit em assets/)

---

## 6. Experimentos e resultados

### 6.1. Cenários de teste

Tabela com os 14 cenários:

| ID | Categoria | Entradas | CHS obtido | Classificação | Coerência |
|---|---|---|---|---|---|
| C01 | Crítico | eng=5, sup=2, sat=3, fin=60, ten=18 | _<rodar>_ | _<rodar>_ | ✅ |
| ... | ... | ... | ... | ... | ... |

(Gerar tabela rodando `pytest tests/ -v --tb=no` e copiando saídas)

### 6.2. Superfícies de controle

Inserir 3-4 superfícies 3D (assets/control_surfaces/):
- engajamento × satisfação
- saude_financeira × engajamento
- volume_suporte × satisfação
- tenure × engajamento

### 6.3. Análise de sensibilidade (trilha de ampliação)

Variar parâmetros de **engajamento_alto** ([55,70,85] → [50,70,90] etc.) e medir variação do CHS médio em 100 clientes sintéticos. Gerar gráfico antes/depois.

### 6.4. Otimização PSO (pontuação extra)

- **Função objetivo:** minimizar erro entre CHS do sistema e CHS rotulado por "especialista" (10 personas)
- **Representação:** vetor com parâmetros das MFs
- **Critério de parada:** 50 iterações ou convergência
- **Resultados:** tabela antes/depois + gráfico de convergência

### 6.5. Discussão crítica

- Onde o sistema funciona bem?
- Onde falha?
- Quais MFs são mais sensíveis?
- Que regras poderiam ser adicionadas?

---

## 7. Conclusão e trabalhos futuros

- Síntese do que foi entregue
- Limitações conhecidas
- Próximos passos: integração com CRMs (HubSpot, Salesforce), módulo de ML para auto-rotular clientes, expansão para mais variáveis (uso de API, eventos de produto)

---

## Declaração de uso de IA

Ver documento separado `docs/declaracao_uso_ia.md`.

---

## Referências

1. Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.
2. Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies*, 7(1), 1-13.
3. Klir, G. J., & Yuan, B. (1995). *Fuzzy sets and fuzzy logic: theory and applications*. Prentice Hall.
4. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of IEEE International Conference on Neural Networks*.
5. _<adicionar pelo menos 5 artigos JCR/Qualis A1-A4 sobre fuzzy aplicado a CHS, churn ou Customer Success>_
6. scikit-fuzzy documentation. https://pythonhosted.org/scikit-fuzzy/
7. Streamlit documentation. https://docs.streamlit.io/

---

## Apêndice A — Base completa de regras

(Inserir tabela das 28 regras com justificativas)

## Apêndice B — Parâmetros completos das MFs

(Inserir tabela com todos os parâmetros de `MFParams`)
