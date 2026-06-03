# HealthSense Allocator — Otimização Evolutiva da Carteira de Customer Success

**Trabalho de Desenvolvimento de Protótipo — Parte 2: IA Evolutiva e Computação Bioinspirada**
**Disciplina:** Inteligência Artificial e Computacional (0700M8) — CESUPA 2026/1
**Professor:** Daniel Leal Souza
**Opção:** 2 (Protótipo de Programa) · **Algoritmo:** Algoritmo Genético
**Ampliação (equipe 5):** Comparação AG vs Memético · **Extra:** Integração Fuzzy–Evolutiva

**Repositório GitHub:** _<inserir link>_

**Equipe:**
- _<Nome 1>_ — _<papel>_
- _<Nome 2>_ — _<papel>_
- _<Nome 3>_ — _<papel>_
- _<Nome 4>_ — _<papel>_
- _<Nome 5>_ — _<papel>_

---

## Resumo

Apresentamos o **HealthSense Allocator**, um protótipo de apoio à decisão que otimiza a alocação da capacidade de um time de Customer Success (CS) em SaaS B2B. Dada uma carteira de clientes com seus sinais de saúde, um conjunto de CSMs com orçamentos de horas e especialidades, e um catálogo de *playbooks* (intervenções), o sistema decide **qual CSM atende qual cliente com qual playbook** de modo a **maximizar a receita esperada retida/expandida**. O problema é formulado como um **Problema de Atribuição Generalizada (GAP)** e resolvido por um **Algoritmo Genético** cuja função de aptidão reusa o **sistema fuzzy de Customer Health Score da Parte 1** para prever o ganho de saúde de cada intervenção. O AG supera consistentemente uma heurística gulosa (+2,1% em média, 5 sementes) e a busca aleatória (+55%). Comparamos ainda o AG com uma versão **Memética** (AG + busca local) e mostramos, com a contagem correta de avaliações da função objetivo, que o ganho aparente do memético por geração é um artefato — no eixo justo o AG puro domina. A integração fuzzy é decisiva: substituí-la por um modelo linear degrada o valor real da alocação em ~64%.

**Palavras-chave:** algoritmo genético, GAP, computação evolutiva, lógica fuzzy, Customer Success.

---

## 1. Problema e público-alvo

### 1.1. Contexto
SaaS B2B vive de receita recorrente; reter e expandir contas é função do time de Customer Success. Esses times têm **capacidade limitada** e carteiras grandes, então precisam priorizar: a cada ciclo, em quais contas investir e com qual ação.

### 1.2. Público-alvo
- Líderes de Customer Success (planejamento de capacidade do time)
- CS Operations (desenho de playbooks e roteamento)
- CSMs (recebem o plano de ação priorizado)

### 1.3. Decisão apoiada
Para cada cliente: **(CSM, playbook)** ou nenhuma ação, respeitando o orçamento de horas de cada CSM, maximizando a receita esperada.

### 1.4. Por que Computação Evolutiva
- Espaço combinatório de ~10⁵⁶ alocações.
- GAP é NP-difícil (múltiplas capacidades + acoplamento por especialidade).
- Objetivo **não-linear/não-diferenciável** (fuzzy) ⇒ sem solução exata trivial; metaheurística é adequada.

---

## 2. Requisitos

### 2.1. Funcionais
| ID | Requisito |
|---|---|
| RF01 | Carregar/gerar carteira com sinais de saúde e MRR |
| RF02 | Otimizar a alocação (CSM, playbook) por cliente |
| RF03 | Respeitar o orçamento de horas de cada CSM |
| RF04 | Exibir receita otimizada, plano de ação e horas por CSM |
| RF05 | Comparar com métodos simples (guloso, aleatório, sem-otimização) |
| RF06 | Exportar o plano de ação (CSV) |

### 2.2. Não funcionais
| ID | Requisito |
|---|---|
| RNF01 | Otimização em poucos segundos para N≤60 |
| RNF02 | Reprodutibilidade por semente |
| RNF03 | Execução via navegador (Streamlit) e via scripts |

### 2.3. Limitações e riscos de interpretação
- ΔCHS é **estimativa** do efeito de um playbook, não garantia.
- Portfólio e efeitos são sintéticos (calibrados por conhecimento de domínio).
- A ferramenta apoia a decisão; não substitui o julgamento do CSM.

---

## 3. Formulação da otimização

### 3.1. Variáveis de decisão
Vetor `x ∈ {0,...,M·K}ᴺ`. Para o cliente *i*: `x_i = 0` (nenhuma ação) ou decodifica em (CSM *j*, playbook *p*).

### 3.2. Representação
Inteiros; `csm = (x_i−1) // K`, `playbook = (x_i−1) % K`. (N=40, M=5, K=5 ⇒ 26 opções/cliente.)

### 3.3. Função objetivo / aptidão
```
maximizar  f(x) = Σ_i MRR_i · (ΔCHS_i / 100)   −   1000 · Σ_j max(0, horas_j − orçamento_j)
```
- `ΔCHS_i = CHS_fuzzy(estado_pós_ação) − CHS_fuzzy(estado_atual)`.
- O efeito do playbook é escalado pela **especialidade** do CSM (multiplicador).
- A penalidade torna inviável estourar orçamento (reforçada pelo operador de reparo).

### 3.4. Restrições e factibilidade
Σ horas por CSM ≤ orçamento do CSM. Garantida por **operador de reparo**: remove as atribuições de pior densidade (valor/hora) de CSMs estourados até caber.

### 3.5. Métricas
- **Principal:** receita esperada (valor da função objetivo).
- **Comparação:** % sobre guloso/aleatório/sem-otimização.
- **Desempenho:** gerações, **nº de avaliações da função objetivo**, tempo, melhor/média/desvio em 5 sementes.

---

## 4. Motor evolutivo

| Componente | Escolha | Parâmetro padrão |
|---|---|---|
| Seleção | Torneio | k = 3 |
| Crossover | Uniforme | taxa = 0,9 |
| Mutação | Reatribuição por gene | taxa = 0,03 |
| Elitismo | Preserva melhores | 2 |
| População | — | 80 |
| Parada | Gerações ou estagnação | 150 ger / paciência 30 |
| Reparo | Densidade | sempre |

**Justificativa do algoritmo:** o AG combina exploração (crossover/mutação) e explotação (seleção/elitismo) sobre representação de atribuição, sem exigir gradiente — adequado ao objetivo fuzzy.

**Versão Memética (ampliação):** após cada geração, busca local (hill-climbing de troca de opção por cliente) refina os 3 melhores indivíduos (aprendizado lamarckiano). Avaliações da busca local são contabilizadas.

---

## 5. Arquitetura e implementação

### 5.1. Componentes
```
Usuário → Dashboard (Streamlit) → GeneticAlgorithm/MemeticAlgorithm
                                        ↓ aptidão
                              AllocationProblem (GAP)
                                        ↓ ΔCHS
                              HealthSenseEngine (fuzzy, 32 regras — Parte 1)
```

### 5.2. Reuso da Parte 1
O motor fuzzy (`src/fuzzy/`) é o da Parte 1, vendorado sem alteração de lógica, usado como **oráculo de aptidão**.

### 5.3. Decisão de engenharia: pré-cômputo
O valor/horas de cada (cliente, opção) é pré-computado uma vez (N×26 avaliações fuzzy). Isso acelera o AG sem trivializar o GAP — a dificuldade está em satisfazer as M restrições de capacidade simultâneas.

---

## 6. Experimentos e resultados

> Todos reproduzíveis: `python experiments/run_experiments.py`. Tabelas em `data/`, figuras em `assets/`.

### 6.1. Validação: AG vs métodos simples (5 sementes)
_(inserir `assets/parte2/exp1_validacao.png` e `data/exp1_validacao.csv`)_

| Método | Receita média (R$) | Atendidos |
|---|---|---|
| Sem otimização | 0 | 0 |
| Busca aleatória | 6.255 | 22 |
| Guloso | 9.517 | 29 |
| **AG** | **9.718 (+2,1%)** | 26 |

A busca aleatória **com reparo** (6.255) fica muito abaixo do guloso: o ganho do AG vem da busca evolutiva, não do reparo. Desvio do AG baixo (~42) ⇒ estável.

### 6.2. Ampliação obrigatória — AG vs Memético
_(inserir `assets/parte2/exp2_ag_vs_memetico.png`)_

| Método | Fitness médio | Avaliações | Tempo |
|---|---|---|---|
| AG | 9.718 | 10.298 | 1,0 s |
| Memético | 9.672 | 41.802 (4,1×) | 3,0 s |

**Interpretação crítica.** No **eixo de gerações** (Fig. esquerda) o memético atinge o platô antes — sedutor, mas enganoso, pois cada geração do memético faz ~4× mais avaliações da função objetivo. No **eixo de avaliações** (Fig. direita), o AG puro atinge a mesma qualidade com ~¼ do custo. Conclusão honesta: para este GAP, o **crossover uniforme já explora a estrutura**, e a busca local de gene único não compensa seu custo. É um resultado negativo informativo: complexidade algorítmica deve ser justificada empiricamente.

> **Ampliação obrigatória cumprida.** A trilha escolhida (Comparação ampliada — versão simples vs híbrida) está atendida: **duas abordagens distintas foram implementadas, comparadas com métricas e gráficos, e discutidas**. O enunciado (Seção 5) não exige que a versão híbrida vença; exige a comparação e a análise — entregues aqui com rigor metodológico (contagem correta de avaliações e eixo justo).

### 6.3. Sensibilidade ao regime de capacidade
_(inserir `assets/parte2/exp3_capacidade.png` e `data/exp3_capacidade.csv`)_

O AG supera o guloso em **todos** os regimes; a magnitude depende da capacidade: menor sob escassez extrema (**~+2%**, poucas escolhas ⇒ guloso quase-ótimo) e crescente com a folga (até **+7,0%**), onde a miopia do guloso custa caro — ele **estagna/regride** (9.974 → 9.825 ao passar de fator 1,2 para 1,6) ao serializar mal as atribuições, enquanto o AG aproveita a capacidade extra (10.511). Defende empiricamente quando o AG "vale a pena".

### 6.4. Extra — impacto da integração Fuzzy–Evolutiva
_(inserir `assets/parte2/exp4_fuzzy_vs_linear.png` e `data/exp4_fuzzy_vs_linear.csv`)_

**O mecanismo (manchete, não rodapé):** o *surrogate* linear não é um espantalho — é a alternativa plausível que uma equipe usaria **sem** o modelo fuzzy: "maximize a soma das melhorias de input, ponderada por MRR". O que ele não enxerga é a **saturação**: um +15 de engajamento numa conta já saudável (CHS≈78) produz ΔCHS≈0, mas o modelo linear, olhando só a magnitude do efeito, gasta horas preciosas ali. O fuzzy captura essa não-linearidade e desvia as horas para contas de fronteira, onde a mesma intervenção move muito o CHS. É exatamente por isso que a integração Fuzzy–Evolutiva agrega valor — e por isso o efeito é grande.

Substituindo a aptidão fuzzy por esse *surrogate* linear:
- **29/40** clientes recebem atribuição diferente.
- Avaliadas no objetivo **verdadeiro** (fuzzy): **fuzzy = 9.747 vs linear = 3.484**.
- O modelo linear desperdiça horas em contas já saudáveis (efeito grande, ΔCHS≈0). O fuzzy direciona as horas a contas onde a intervenção realmente move o ponteiro. A integração não é decorativa: **é o que torna a alocação boa.**

### 6.5. Limitações
- Dados sintéticos; calibração de efeitos por domínio, não medida em campo.
- Busca local simples (gene único); operadores de troca mais ricos poderiam mudar a conclusão 6.2.
- Penalidade/reparo assumem que toda conta aceita no máximo uma ação por ciclo.

---

## 7. Conclusão e trabalhos futuros

O protótipo demonstra uma aplicação coerente de Computação Evolutiva a um problema real de CS, com a lógica fuzzy como núcleo de avaliação. O AG entrega alocações melhores que heurísticas simples, de forma estável e reprodutível, num produto demonstrável. **Trabalhos futuros:** operadores de troca/realocação na busca local; múltiplas ações por conta; otimização multiobjetivo (receita × esforço × risco); integração com CRM real; *warm-start* incremental entre ciclos.

---

## Declaração de uso de IA
Ver `docs/declaracao_uso_ia.md`.

## Referências
1. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*.
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*.
3. Moscato, P. (1989). On Evolution, Search, Optimization, GAs and Martial Arts (Memetic Algorithms).
4. Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*.
5. Mamdani, E. H. (1975). An experiment in linguistic synthesis with a fuzzy logic controller.
6. _<adicionar referências de GAP / metaheurísticas para alocação de recursos>_
7. scikit-fuzzy; Streamlit (documentação).

## Apêndices
- **A.** Catálogo de playbooks e matriz de especialidades (de `src/evolutionary/problem.py`).
- **B.** Base de 32 regras fuzzy (de `src/fuzzy/rules.py`).
- **C.** Tabelas completas dos experimentos (`data/*.csv`).
