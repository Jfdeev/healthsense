# Declaração de Uso de IA — HealthSense (Partes 1 e 2)

**Equipe:** _<preencher nomes>_
**Disciplina:** Inteligência Artificial e Computacional (0700M8) — CESUPA 2026/1
**Professor:** Daniel Leal Souza

Esta declaração atende ao item de **Declaração de uso de IA** dos enunciados das Partes 1 e 2. A equipe usou ferramentas de IA com transparência e revisão crítica; **todo o código, texto e experimentos foram revisados, executados e validados pelos integrantes**.

## Ferramentas e usos

| Ferramenta | Parte | Finalidade | Revisão crítica da equipe |
|---|---|---|---|
| Claude (Anthropic) | 1 | Arquitetura do projeto fuzzy, variáveis e termos linguísticos, esboço da base de regras e do dashboard | Aceito após cross-check com literatura de Customer Success; faixas e parâmetros calibrados pela equipe; regras de conflito adicionadas pela equipe após observar saídas inadequadas |
| Claude (Anthropic) | 1 | Sugestão de defuzzificação e extensão PSO | Equipe pesquisou a justificativa do centróide antes de adotar; rejeitou MOM e a 6ª variável sugeridas |
| Claude (Anthropic) | 2 | Formular o problema de CS como GAP e desenhar o motor evolutivo | Equipe **rejeitou** a primeira formulação (separável) e adotou o GAP acoplado por especialidades |
| Claude (Anthropic) | 2 | Esboço do AG, do operador de reparo e do memético | Validado empiricamente: sem reparo o AG perdia para o guloso; a comparação AG vs Memético foi corrigida para contar as avaliações da busca local |
| GitHub Copilot | 1 e 2 | Autocomplete durante a implementação | Cada sugestão revisada antes de aceitar |
| ChatGPT / Gemini | 1 e 2 | Revisão de texto de relatórios e slides | Mantido o argumento da equipe quando a IA descaracterizava |

> Atualizar/substituir conforme o uso real da equipe.

## Princípios seguidos
1. **Transparência** — toda contribuição relevante da IA está documentada.
2. **Revisão humana** — nada entrou no repositório sem leitura e validação do integrante responsável.
3. **Validação empírica** — toda afirmação de desempenho foi medida (5 sementes, métricas reproduzíveis), não aceita por sugestão.
4. **Honestidade metodológica** — quando os dados contrariaram uma hipótese conveniente (ex.: memético "mais rápido"), reportamos o resultado honesto.
5. **Sem cópia literal** — explicações reescritas na voz da equipe; código adaptado e comentado pelos integrantes.

## Responsabilidade
A equipe assume integral responsabilidade pelo conteúdo final — código, relatórios, experimentos e slides das duas partes.

## Quem revisou o quê
| Integrante | Revisou |
|---|---|
| _<nome>_ | `src/fuzzy/` (Parte 1) |
| _<nome>_ | `src/evolutionary/problem.py`, `ga.py` (Parte 2) |
| _<nome>_ | `memetic.py`, experimentos |
| _<nome>_ | `app/` (dashboard) |
| _<nome>_ | testes, métricas, relatórios |
