# Resumo Conceitual — HealthSense (base para apresentar e defender)

Material de estudo para os 5 integrantes. Objetivo: entender os **conceitos** e os **porquês**, não decorar. Lê-se em ~20 min.

---

## 0. A visão de uma frase

> **HealthSense é uma plataforma de Customer Success para SaaS B2B. O Módulo 1 (fuzzy) ENXERGA a saúde de cada cliente; o Módulo 2 (evolutivo) decide como AGIR com recursos limitados — e usa o Módulo 1 como cérebro de avaliação.**

Dois trabalhos da disciplina, um produto só:
- **Parte 1** = Sistemas de Controle Fuzzy → o **Customer Health Score (CHS)**.
- **Parte 2** = IA Evolutiva → o **Allocator** (otimização da carteira).

---

# PARTE 1 — Sistema Fuzzy (Customer Health Score)

## 1.1. O problema e por que fuzzy

Customer Success precisa priorizar a carteira todo dia. Os jeitos tradicionais falham:
- **Score binário** (bom/ruim) → perde a nuance.
- **Soma de pesos** (ex.: 0,3×uso + 0,2×NPS…) → pesos arbitrários, não capturam interações.
- **Regras rígidas if/else** → quebram na fronteira (atraso de 30 dias = ok, 31 = crítico?).

**Lógica fuzzy** resolve porque permite **graus** de verdade (não só 0 ou 1), **transições suaves** e **regras em linguagem natural** que combinam variáveis como um especialista pensaria.

## 1.2. Conceitos fundamentais de lógica fuzzy

- **Conjunto fuzzy:** diferente do conjunto clássico (onde um elemento pertence ou não), aqui cada elemento tem um **grau de pertinência** entre 0 e 1. Ex.: um cliente com engajamento 62 pode pertencer 0,7 a "Alto" e 0,3 a "Moderado".
- **Variável linguística:** uma variável cujos valores são *palavras* — "engajamento" assume "Baixo/Moderado/Alto/Muito Alto".
- **Termo linguístico:** cada palavra (cada conjunto fuzzy) de uma variável.
- **Universo de discurso:** a faixa numérica da variável (ex.: engajamento ∈ [0, 100]).
- **Função de pertinência (MF):** a função que mapeia um número ao grau de pertinência em um termo. É o coração da modelagem.

## 1.3. As variáveis do HealthSense (5 entradas, 1 saída)

| Variável | Universo | Termos |
|---|---|---|
| Engajamento | 0–100 | Baixo, Moderado, Alto, Muito Alto |
| Volume de suporte | 0–30 tickets | Saudável, Atenção, Alto, Crítico |
| Satisfação (NPS) | 0–10 | Detrator, Neutro, Promotor |
| Saúde financeira | 0–90 dias atraso | Em dia, Atraso leve, Atraso grave |
| Tenure | 0–60 meses | Novo, Estabelecido, Veterano |
| **Saída: CHS** | 0–100 | Crítico, Em Risco, Atenção, Saudável, Promotor |

**Por que essas 5?** São os pilares de um health score real: uso (engajamento), fricção (suporte), percepção (NPS), compromisso (financeiro), maturidade (tenure). Cada uma tem papel distinto.

## 1.4. Tipos de função de pertinência (e por que misturamos)

- **Trapezoidal** (4 parâmetros): platô no topo. Usada nos **extremos** (ex.: "Baixo", "Crítico") — valores muito baixos/altos devem ter pertinência plena 1 numa faixa, não num único ponto.
- **Triangular** (3 parâmetros): um pico. Usada nos **termos intermediários** — têm um valor central de referência.
- **Gaussiana** (média, desvio): curva suave. Usada na **saída (CHS)** — garante superfície de controle contínua e defuzzificação estável.

## 1.5. Como o sistema decide (inferência Mamdani)

O caminho de uma entrada até o score (decore esse fluxo de 5 passos):

1. **Fuzzificação:** cada valor numérico vira graus de pertinência nos termos.
2. **Avaliação das regras:** o "E" (AND) entre condições = **mínimo** das pertinências.
3. **Implicação:** o grau de ativação da regra **recorta** (mínimo) o conjunto de saída.
4. **Agregação:** combinam-se todas as regras ativadas com **máximo** (união).
5. **Defuzzificação:** o **centróide** (centro de massa) da área agregada vira o CHS final.

> **AND = min, OR = max, implicação = min, agregação = max, defuzzificação = centróide.** Isso é "Mamdani".

**Por que centróide?** A saída é um número (um score), e o centróide dá uma saída **contínua e suave** — pequenas mudanças na entrada → pequenas mudanças no score. Alternativas como MOM (média dos máximos) dariam saltos.

## 1.6. A base de 32 regras

Agrupadas em 7 categorias com justificativa de domínio:
- **Crítico (4)**, **Em Risco (6)**, **Atenção (6)**, **Saudável (5)**, **Promotor (4)**
- **Conflito (3):** exceções para variáveis que se opõem. Ex.: *power user inadimplente* (usa muito mas não paga) → rebaixado, porque uso não compensa risco financeiro.
- **Baseline (4):** garantem que **qualquer** entrada ative ao menos uma regra (sem "buracos" no espaço).

## 1.7. Mamdani vs TSK (pode cair na arguição)

- **Mamdani** (o nosso): o consequente da regra é um **conjunto fuzzy** ("CHS é Saudável") → precisa defuzzificar. Mais interpretável.
- **TSK** (Takagi-Sugeno-Kang): o consequente é uma **função** (ex.: y = a₀ + a₁x₁) → saída por média ponderada, sem defuzzificação. Melhor para controle fino, menos interpretável.
Escolhemos Mamdani porque é o padrão da modalidade e é mais explicável para um CSM.

## 1.8. Validação e extras (Parte 1)

- **14 cenários** (baixos, médios, altos, fronteiriços, conflitantes) + **16 testes** automatizados.
- **Superfícies de controle:** gráficos 3D mostrando o CHS variar por duas variáveis — evidência de que o sistema é coerente e suave.
- **Ampliação (equipe de 5):** análise de **sensibilidade** — deslocar o centro de uma MF e medir o impacto no CHS médio de 100 personas. Mostra que a calibração importa.
- **Extra (0,5): PSO** (Particle Swarm Optimization) ajusta automaticamente os parâmetros das MFs para casar com rótulos de especialista.

---

# PARTE 2 — Sistema Evolutivo (Allocator)

## 2.1. O problema e por que computação evolutiva

O time de CS tem **horas limitadas**. Decisão: qual CSM atende qual cliente, com qual **playbook** (intervenção), para maximizar a receita retida/expandida.

**Por que não uma conta simples ou programação linear?**
1. **Espaço gigante:** 40 clientes × 26 opções cada = 26⁴⁰ ≈ 10⁵⁶ combinações.
2. **NP-difícil:** é um **Problema de Atribuição Generalizada (GAP)** — várias restrições de capacidade (uma por CSM) e a eficácia depende do trio (cliente, CSM, playbook) via **especialidades**. As decisões ficam acopladas.
3. **Objetivo não-linear:** o ganho de saúde vem do **fuzzy** (min/max, regras), que não é linear nem derivável → programação linear não serve.

Metaheurísticas (como o Algoritmo Genético) brilham nesse cenário: objetivo "caixa-preta", espaço combinatório.

## 2.2. Conceitos de Computação Evolutiva

Inspiração na **seleção natural**: uma população de soluções "evolui" ao longo de gerações; as melhores se reproduzem e sofrem variação.

- **Indivíduo / cromossomo:** uma solução candidata (no nosso caso, um plano de alocação).
- **População:** conjunto de indivíduos.
- **Função de aptidão (fitness):** mede quão boa é a solução (aqui: a receita esperada).
- **Seleção:** escolhe os "pais" — melhores têm mais chance.
- **Crossover (recombinação):** combina dois pais para gerar filhos.
- **Mutação:** pequena alteração aleatória → mantém diversidade.
- **Elitismo:** preserva os melhores entre gerações (não perde a melhor solução).
- **Critério de parada:** nº de gerações ou estagnação.

## 2.3. Como modelamos (formulação)

- **Representação:** um vetor de inteiros de tamanho N. Cada posição (cliente) guarda a opção: 0 = nenhuma ação, ou um código que decodifica em (CSM, playbook).
- **Função de aptidão:** `f(x) = Σ MRR × (ΔCHS/100) − penalidade por horas excedidas`.
  - `MRR` = receita mensal do cliente; `ΔCHS` = ganho de saúde previsto **pelo fuzzy**.
- **Restrições:** soma de horas de cada CSM ≤ seu orçamento.
- **Factibilidade (operador de reparo):** se um CSM estoura o orçamento, removemos as ações de **pior densidade** (valor/hora) até caber. Mantém toda a população válida — foi decisivo: **sem reparo, o AG perdia para o guloso.**

## 2.4. O motor (parâmetros que usamos)

Seleção por **torneio** (k=3) · **crossover uniforme** · **mutação** por gene (~3%) · **elitismo** (2) · população 80 · parada por estagnação. 

- **Por que torneio?** Pressão seletiva controlável e robusto a escalas de fitness.
- **Por que crossover uniforme?** Cada cliente é uma posição independente; trocar gene a gene mistura bem boas atribuições dos dois pais.

## 2.5. A integração Fuzzy–Evolutiva (o coração + extra 0,5)

A função de aptidão **é** o motor fuzzy da Parte 1:

```
AG → "e se eu aplicar este playbook neste cliente?" → Fuzzy CHS (32 regras) → ΔCHS → valor (R$)
```

**Prova de que não é decoração:** trocamos o fuzzy por um modelo **linear** (que ignora a saturação do CHS). Resultado: **29 de 40** alocações mudam, e o valor real cai de **9.747 para 3.484**. Por quê? O linear desperdiça horas em contas já saudáveis (onde o ganho real é ≈0); o fuzzy enxerga essa saturação e manda as horas para quem realmente se beneficia.

## 2.6. Comparações e resultados (Parte 2)

**Baselines obrigatórios** (5 sementes, receita média R$):
- Sem otimização: 0 · Busca aleatória: 6.255 · Guloso: 9.517 · **AG: 9.718** (+2,1% vs guloso, +55% vs aleatório)
- A aleatória *com reparo* fica bem abaixo do guloso → prova que o ganho vem da **busca evolutiva**, não do reparo.

**Ampliação (AG vs Memético)** — o ponto mais importante para a defesa:
- Memético = AG + busca local refinando a elite.
- No eixo de **gerações**, o memético parece convergir antes — **mas é uma ilusão**: cada geração dele custa ~4× mais avaliações da função objetivo.
- No eixo de **avaliações** (o justo), o **AG puro domina** (mesma qualidade, ¼ do custo).
- **Conclusão honesta:** o crossover já explora bem a estrutura; a busca local de gene único não compensa. Resultado negativo, mas **cientificamente válido** — e é exatamente o que a ampliação pede (comparar e discutir).

**Sweep de capacidade:** a vantagem do AG sobre o guloso cresce com a folga de capacidade (de ~+2% até +7%), porque a miopia do guloso custa mais caro quando há mais opções.

---

# 3. A história para contar (fio condutor)

1. **Dor real:** times de CS precisam priorizar carteiras grandes com pouco tempo.
2. **Enxergar (Parte 1):** o fuzzy traduz 5 sinais em um score interpretável, com regras que um humano entende.
3. **Agir (Parte 2):** o AG transforma o score em decisão — onde investir as horas para maximizar receita.
4. **A sacada:** o mesmo motor fuzzy que pontua também avalia as ações — integração Fuzzy–Evolutiva.
5. **Honestidade:** validamos com baselines, 5 sementes e métricas corretas; até reportamos um resultado negativo (memético) por integridade.

---

# 4. Conceitos que VÃO cair na arguição (checklist)

**Parte 1:** o que é grau de pertinência · por que 3 tipos de MF · os 5 passos da inferência Mamdani · por que centróide e não MOM · diferença Mamdani × TSK · o que é uma regra de conflito.

**Parte 2:** o que é fitness/seleção/crossover/mutação · por que o problema é NP-difícil (GAP) · por que AG e não programação linear/guloso · o que o operador de reparo faz · por que a comparação AG×Memético deve ser no eixo de avaliações · como o fuzzy entra como aptidão.

> Detalhes em pergunta-resposta: `docs/roteiro_defesa_parte1.md` e `docs/roteiro_defesa_parte2.md`.
