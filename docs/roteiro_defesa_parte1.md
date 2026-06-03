# Roteiro de Defesa — Parte 1 (HealthSense Fuzzy)

Guia de arguição. **Todos os 5 integrantes** devem conseguir responder qualquer item — o professor pode direcionar a pergunta a qualquer um. Estude os "porquês", não decore.

---

## Pitch de 60 segundos (abertura)

> "Customer Success em SaaS B2B precisa priorizar a carteira todo dia: quem está prestes a cancelar, quem pode crescer. Scores tradicionais são binários ou somas de pesos arbitrários que quebram na fronteira. Nosso **HealthSense** usa lógica fuzzy Mamdani para combinar 5 sinais do cliente em um Customer Health Score de 0 a 100, com transições suaves e regras que explicam cada decisão. É entregue como dashboard interativo, validado com 14 cenários e 16 testes."

---

## Perguntas prováveis e respostas

### Sobre o problema e a escolha do fuzzy

**P: Por que lógica fuzzy e não um modelo de machine learning?**
R: Três razões. (1) **Explicabilidade** — cada score é justificado pelas regras ativadas, essencial para um CSM confiar e agir; ML seria caixa-preta. (2) **Não precisamos de dados rotulados** — fuzzy codifica conhecimento de domínio diretamente. (3) **Transições suaves e interações** — captura "atraso de 30 ≈ 31 dias" e combinações de variáveis sem treinar nada. ML viria depois, quando houvesse histórico de churn rotulado.

**P: Por que essas 5 variáveis?**
R: São os pilares de um Customer Health Score na prática de mercado (Gainsight, HubSpot): **uso do produto** (engajamento), **fricção** (suporte), **percepção** (NPS), **compromisso financeiro** (atraso) e **maturidade do relacionamento** (tenure). Cada uma tem papel distinto; evitamos variáveis redundantes.

**P: Por que o tenure importa?**
R: Porque o mesmo sinal significa coisas diferentes conforme a idade da conta. Um cliente novo com engajamento moderado está em onboarding normal; um veterano com engajamento baixo é alerta de não-renovação. O tenure modula a interpretação das outras variáveis (regras R5, R9, R11, R16…).

### Sobre a modelagem

**P: Por que misturou funções de pertinência trapezoidais, triangulares e gaussianas?**
R: Cada tipo reflete a natureza do termo. **Trapezoidal** nos extremos (ex.: engajamento "Baixo") porque valores muito baixos devem ter pertinência plena 1 numa faixa, não num único ponto. **Triangular** nos intermediários porque têm um valor central de referência claro. **Gaussiana** na saída (CHS) para garantir superfície de controle suave e defuzzificação por centróide estável.

**P: Como você define os parâmetros das funções de pertinência? Não são arbitrários?**
R: Partem do domínio (ex.: NPS 0–6 detrator, 7–8 neutro, 9–10 promotor é a definição clássica). E não ficamos só na intuição: a **análise de sensibilidade** (ampliação) mostra o quanto o sistema muda quando deslocamos uma MF, e o **PSO** (extra) ajusta os parâmetros automaticamente contra rótulos de especialista. Ou seja, calibramos com método.

**P: Por que defuzzificação por centróide e não MOM (Mean of Maximum)?**
R: Porque a saída é um **número** (score), e o centróide produz uma saída contínua e suave — pequenas mudanças na entrada geram pequenas mudanças no score. O MOM dá saltos (escolhe o "platô" de máximo), inadequado para um índice. Testamos e o centróide foi mais estável.

**P: Por que 32 regras? Não é muita/pouca?**
R: Cobrem casos típicos, intermediários, **críticos e de conflito**, mais 4 regras *baseline* que garantem que qualquer entrada ative ao menos uma regra (sem buracos). Não é quantidade mecânica: cada regra tem justificativa de Customer Success. As de conflito (R26–R28) surgiram de observar saídas inadequadas nos testes.

**P: O que é uma "regra de conflito"? Dê um exemplo.**
R: É uma exceção que evita saída absurda quando variáveis se opõem. Ex.: R26 — um *power user* (engajamento muito alto) mas **inadimplente grave**. O uso intenso não compensa o risco financeiro, então a regra rebaixa o cliente para "Em Risco" em vez de deixá-lo como saudável.

### Sobre inferência

**P: Explique o caminho de uma entrada até a saída.**
R: (1) **Fuzzificação**: cada valor numérico vira graus de pertinência nos termos. (2) **Avaliação das regras**: AND = mínimo entre as pertinências do antecedente. (3) **Implicação**: o grau de ativação recorta a MF do consequente (mínimo). (4) **Agregação**: combinamos as saídas de todas as regras com máximo. (5) **Defuzzificação**: centróide da área agregada → o CHS final.

**P: O que acontece se nenhuma regra disparar?**
R: Não acontece — as 4 regras *baseline* (R29–R32) garantem cobertura por engajamento. Mas há ainda um *fallback* defensivo no código que retorna 50 (neutro) caso o motor não produzisse saída, para nunca quebrar.

### Sobre validação e limitações

**P: Como você sabe que o sistema funciona?**
R: 14 cenários cobrindo casos baixos, médios, altos, fronteiriços e conflitantes, mais 16 testes automatizados (pytest) — todos passam. Validamos comportamento, não só saídas pontuais: superfícies de controle mostram que o CHS varia de forma coerente e suave entre as variáveis.

**P: Quando o sistema falha ou é frágil?**
R: É sensível à calibração das MFs (mostramos isso na sensibilidade). Os parâmetros refletem práticas de mercado, não dados reais de uma empresa específica — então requer recalibração por contexto. E é apoio à decisão, não diagnóstico determinístico: o CSM continua no comando.

**P: Qual a relação com a Parte 2?**
R: Este motor fuzzy vira a **função de aptidão** do otimizador evolutivo da Parte 2 — ele avalia "se eu fizer tal intervenção, quanto a saúde do cliente melhora?". Um produto, duas capacidades: pontuar e agir.

---

## Divisão sugerida na apresentação (todos falam)

| Integrante | Bloco |
|---|---|
| 1 | Problema + por que fuzzy |
| 2 | Variáveis + funções de pertinência |
| 3 | Base de regras + inferência Mamdani |
| 4 | Demo do dashboard ao vivo |
| 5 | Validação + ampliação (sensibilidade) + extra (PSO) |

**Regra de ouro:** quem não apresentar um bloco ainda precisa saber respondê-lo — o professor escolhe quem responde.
