# Roteiro de Defesa — Parte 2 (HealthSense Allocator)

Guia de arguição. **Todos os 5 integrantes** devem dominar — a apresentação vale **0,60** (o maior peso) e o professor pode perguntar a qualquer um. Foque nos "porquês".

---

## Pitch de 60 segundos (abertura)

> "Um time de Customer Success tem horas limitadas e uma carteira grande. Quem atender, com qual ação? Modelamos isso como um **Problema de Atribuição Generalizada** — qual CSM atende qual cliente com qual playbook — e resolvemos com um **Algoritmo Genético** cuja função de aptidão é o **sistema fuzzy da Parte 1**, que prevê o ganho de saúde de cada intervenção. O AG supera a heurística gulosa em 2,1% e a busca aleatória em 55%, de forma estável em 5 execuções, entregue como dashboard que gera o plano de ação."

---

## Perguntas prováveis e respostas

### Sobre o problema e a escolha do algoritmo

**P: Por que um Algoritmo Genético e não uma solução exata (programação linear/dinâmica)?**
R: Por dois motivos que se somam. (1) É um **GAP — NP-difícil**: múltiplas restrições de capacidade (uma por CSM) e o valor de uma ação depende do trio (cliente, CSM, playbook) por causa das especialidades; isso acopla as decisões. (2) O objetivo é **não-linear e não-diferenciável** — o ganho de saúde vem de um sistema fuzzy com operadores min/max e base de regras. Programação linear exige objetivo linear; não temos isso. O AG lida com objetivo caixa-preta e espaço combinatório naturalmente.

**P: (A pergunta-armadilha) Se você pré-computa o valor de cada (cliente, opção), isso não vira uma mochila que o guloso/DP resolvem?**
R: Não, e essa foi uma decisão consciente de modelagem. O pré-cômputo só acelera a avaliação — a dificuldade está em satisfazer **as 5 restrições de capacidade simultâneas** com o acoplamento das especialidades. Isso é GAP, que continua NP-difícil mesmo com os valores tabelados. Tanto que o **guloso, que é exatamente a heurística de mochila, perde para o AG** — e perde mais quando há folga de capacidade.

**P: Por que GA e não PSO (que vocês usaram na Parte 1)?**
R: Porque aqui o espaço é **combinatório/discreto** (atribuições inteiras), terreno natural do AG com crossover e mutação sobre vetores. PSO é nativo de espaços contínuos (por isso o usamos para calibrar parâmetros de MFs na Parte 1). Usar a ferramenta certa para cada problema.

### Sobre a formulação

**P: Qual é a representação da solução?**
R: Um vetor de inteiros de tamanho N (clientes). Cada gene codifica a opção do cliente: 0 = nenhuma ação, ou um índice que decodifica em (CSM, playbook) via divisão/módulo pelo número de playbooks.

**P: Qual a função de aptidão exatamente?**
R: `f(x) = Σ MRR_i × (ΔCHS_i/100) − 1000 × horas_excedidas`. O ΔCHS_i é a diferença entre o CHS fuzzy do estado pós-ação e o atual. Ou seja, **a receita esperada retida/expandida**, penalizando qualquer estouro de orçamento.

**P: Como você garante que as soluções respeitam a capacidade?**
R: Com um **operador de reparo**: para cada CSM acima do orçamento, removemos as atribuições de pior densidade (valor/hora) até caber. Toda a população fica factível, então o AG otimiza valor diretamente na região viável. Foi essencial — sem reparo, o AG perdia para o guloso.

**P: O que são as "especialidades" dos CSMs?**
R: Um multiplicador de eficácia: um especialista em Retenção rende mais num Business Review (1,5×) e menos num Onboarding (0,7×). Isso faz o valor depender do par cliente-CSM, acoplando as decisões e tornando o problema genuinamente difícil.

### Sobre os operadores

**P: Por que crossover uniforme e seleção por torneio?**
R: **Uniforme** porque cada posição do vetor (cada cliente) é independente — trocar gene a gene mistura bem boas atribuições dos dois pais. **Torneio** porque dá pressão seletiva controlável (pelo tamanho k) e é robusto a escalas de aptidão, sem precisar normalizar.

**P: Como escolheu os parâmetros (população, taxas)?**
R: População 80, crossover 0,9, mutação 0,03, elitismo 2, parada por estagnação. Valores típicos da literatura; verificamos que convergem de forma estável (desvio ~42 em 5 sementes). Um estudo de sensibilidade desses parâmetros é trabalho futuro.

### Sobre a ampliação (AG vs Memético) — PONTO CRÍTICO

**P: O que é o Memético e o que vocês concluíram?**
R: Memético = AG + busca local (hill-climbing) refinando a elite a cada geração. Comparamos as duas versões e chegamos a um **resultado honesto e contra-intuitivo**: no eixo de *gerações* o memético parece convergir antes, mas suas gerações custam ~4× mais avaliações da função objetivo. **No eixo de avaliações — que é o justo — o AG puro domina**: mesma qualidade com ¼ do custo.

**P: Então sua ampliação "falhou"?**
R: Não. A ampliação obrigatória pede **implementar e comparar duas abordagens e discutir as diferenças** — fizemos isso com rigor. A conclusão de que o híbrido não compensa **é** um resultado científico válido e mais valioso que um ganho forçado: mostra que o crossover uniforme já explora bem a estrutura, e que complexidade algorítmica precisa ser justificada empiricamente. Medir no eixo errado (gerações) teria nos enganado.

**P: Por que a busca local não ajudou?**
R: Porque ela faz movimentos de **gene único** (mudar a opção de um cliente), que são em grande parte subsumidos pela mutação + seleção do próprio AG. As melhorias reais exigiriam movimentos coordenados (trocas entre clientes), que o crossover já provê. Operadores de troca na busca local poderiam mudar isso — é trabalho futuro.

### Sobre a integração Fuzzy-Evolutiva (extra) e comparação

**P: Como o fuzzy entra no otimizador? Não é só enfeite?**
R: É o **núcleo da avaliação**: para cada candidato a ação, aplicamos o efeito do playbook ao estado do cliente e perguntamos ao fuzzy o novo CHS; o ΔCHS é o valor. Provamos que não é decorativo: trocando o fuzzy por um **modelo linear** que ignora a saturação, **29 de 40** alocações mudam e o valor real cai de 9.747 para 3.484. O linear desperdiça horas em contas já saudáveis (onde o ganho real é ≈0); o fuzzy enxerga essa saturação e redireciona.

**P: Como compara o AG com os métodos simples?**
R: Em 5 sementes: sem-otimização 0, aleatória 6.255, guloso 9.517, **AG 9.718** (+2,1% vs guloso, +55% vs aleatório). Mantemos a **busca aleatória com reparo** na comparação de propósito: como ela fica bem abaixo do guloso, prova que o ganho do AG vem da busca evolutiva, e não apenas do operador de reparo.

**P: 2,1% sobre o guloso é pouco. Vale a pena?**
R: Depende do regime, e medimos isso. Sob escassez extrema o guloso é quase-ótimo; mas conforme a capacidade cresce, a miopia do guloso custa caro e a vantagem do AG sobe até **+7%** — e o guloso chega a regredir. Em receita recorrente, alguns por cento ao mês são significativos. E o AG é uma base que escala para objetivos mais ricos (multiobjetivo) onde o guloso não serve.

### Sobre métricas e reprodutibilidade

**P: Quais métricas vocês usaram e por quê?**
R: A principal é o valor da função objetivo (receita esperada). Comparação com baselines (% de ganho). Desempenho do motor: gerações, **número de avaliações da função objetivo**, tempo, e melhor/média/desvio em 5 execuções com sementes distintas. As avaliações da busca local são contabilizadas — é o que torna a comparação AG vs Memético honesta.

**P: É reproduzível?**
R: Sim. `python experiments/parte2_experimentos.py` regenera todas as figuras e tabelas; `pytest tests/` roda os testes; tudo fixado por semente.

---

## Divisão sugerida na apresentação (todos falam)

| Integrante | Bloco |
|---|---|
| 1 | Ponte com a Parte 1 + problema + público-alvo |
| 2 | Formulação (GAP) + por que evolutivo |
| 3 | Motor (operadores, reparo) + integração fuzzy |
| 4 | Demo do dashboard ao vivo (gerar um plano) |
| 5 | Resultados + ampliação (AG vs Memético) + extra (fuzzy vs linear) |

**Regra de ouro:** todos precisam saber explicar a pergunta-armadilha do GAP e o resultado honesto do memético — são os dois pontos onde um professor de Computação Evolutiva vai cutucar.
