# Tila Health Triage — Sistema de Classificação de Risco Clínico

**Trabalho Prático da Disciplina:** Tópicos Avançados em Inteligência Artificial  
**Docente:** Prof. Dr. Márcio Andrey Teixeira  
**Instituição:** Instituto Federal de São Paulo (IFSP) — Campus Catanduva  
**Discentes / Autores:** Ryan Aguiar e Pedro Henrique Oliveira Pereira  
**Data:** Outubro de 2026  

---

## 🌐 Link da Aplicação em Produção
* **Streamlit Cloud:** [https://tila-health-triage.streamlit.app/](https://tila-health-triage.streamlit.app/)
* **Repositório GitHub:** [https://github.com/RyanAguiiarr/tila-health-triage](https://github.com/RyanAguiiarr/tila-health-triage)

---

## 1. Descrição do Problema e Objetivo
Em hospitais e unidades de pronto atendimento, a triagem inicial é essencial para organizar a fila de pacientes por ordem de gravidade e não apenas por ordem de chegada. O **Tila Health Triage** foi desenvolvido como projeto da disciplina para apoiar a equipe de acolhimento e enfermagem nessa decisão.

Com base em 9 parâmetros universais (idade, pressão sistólica, frequência cardíaca, frequência respiratória, saturação de oxigênio, temperatura, nível de dor, histórico de doenças crônicas, idas prévias e forma de chegada), o modelo prevê a prioridade do atendimento em três níveis clínicos:
* **Verde (0 — Baixa Prioridade / Pouco Urgente):** Estável, atendimento em até 120 minutos.
* **Amarelo (1 — Média Prioridade / Urgência Relativa):** Dor moderada ou febre, atendimento em até 60 minutos.
* **Vermelho (2 — Alta Prioridade / Emergência Imediata):** Risco de choque ou insuficiência respiratória, atendimento imediato (zero minutos).

---

## 2. Estrutura dos Arquivos do Projeto
```text
Tila_Triage_Geral/
├── dataset/
│   └── triagem_kaggle_bruto.csv       # Base de dados original do Kaggle (18.000 atendimentos)
├── modelo/
│   └── triagem_model.pkl              # Modelo serializado com o scaler para deploy
├── figuras/
│   ├── matriz_confusao_regressao_logistica.png
│   ├── matriz_confusao_arvore_decisao.png
│   ├── importancia_atributos_dt.png
│   └── comparacao_modelos.png
├── treinamento.py                     # Script de preparação dos dados, treino e avaliação
├── app.py                             # Aplicação interativa em Streamlit
├── requirements.txt                   # Dependências do projeto
├── README.md                          # Instruções de execução e documentação
└── Relatorio_Trabalho1_Tila_Triage.pdf # Relatório técnico oficial em PDF (5 páginas ABNT)
```

---

## 3. Instruções de Execução Local

O projeto foi desenvolvido em Python e segue o passo a passo ensinado nas aulas pelo professor:

### 1. Criar o Ambiente Virtual (.venv)
```bash
python -m venv .venv
```

### 2. Ativar o Ambiente Virtual
* **No Windows:**
  ```cmd
  .venv\Scriptsctivate
  ```
* **No Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Instalar as Dependências
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar o Treinamento e Avaliação
O script carrega os 18.000 registros, faz o pré-processamento, divide em 75% treino / 25% teste, treina a Regressão Logística e a Árvore de Decisão, gera os gráficos em `figuras/` e salva o modelo vencedor em `modelo/triagem_model.pkl`:
```bash
python treinamento.py
```

### 5. Iniciar o Aplicativo Streamlit
```bash
streamlit run app.py
```
A aplicação abrirá no navegador no endereço `http://localhost:8501`.

---

## 4. Resultados Obtidos no Conjunto de Teste (4.500 Amostras)

| Métrica | Regressão Logística (Multiclasse) | Árvore de Decisão (Entropia, depth=6) |
| :--- | :---: | :---: |
| **Acurácia Global** | **96,22%** | 94,18% |
| **F1-Score Macro** | **0,9555** | 0,9406 |
| **Recall (Emergência / Vermelho)** | **93,27%** | 92,83% |
| **Recall (Média Prioridade)** | **94,65%** | 91,31% |
| **Recall (Baixa Prioridade)** | **97,62%** | 96,37% |

* **Modelo Escolhido para o Deploy:** **Regressão Logística Multiclasse**.  
  *Justificativa:* Apresentou a maior acurácia global e o maior Recall na classe de emergência (93,27% contra 92,83%), minimizando os casos graves não detectados, além de fornecer probabilidades contínuas para a interface do usuário.

---

## 5. Casos de Teste Documentados
Os 5 casos de teste obrigatórios foram executados e analisados em detalhes na Seção 6 do Relatório em PDF:
1. **Entrada Válida 1 (Baixa / Verde):** Jovem com sinais estáveis e dor leve -> Previsto como Verde (99,86% de certeza).
2. **Entrada Válida 2 (Média / Amarelo):** Adulto com dor nível 7 e febre de 38,8°C -> Previsto como Amarelo (99,37% de certeza).
3. **Entrada Válida 3 (Alta / Vermelho):** Idoso em crise hipertensiva severa e SpO₂ de 88% -> Previsto como Vermelho (99,98% de certeza).
4. **Entrada Inválida:** PAS de 45 mmHg e Temperatura de 43,5°C -> Interceptado com sucesso pela tela, orientando a conferir os dados.
5. **Caso Limítrofe:** Paciente em fronteira com dor 8 e saturação de 91,5% -> Previsto como Amarelo (73,15%) com alerta de Vermelho (26,85%).

---

## 6. Referências
1. **Dataset do Kaggle:** AKKUŞ, Emirhan. *Synthetic Medical Triage Priority Dataset*, 2024.
2. **Ministério da Saúde:** *Acolhimento à Demanda Espontânea nos Serviços de Urgência*. Cadernos de Atenção Básica, n. 28, Brasília, 2013.
3. **Documentação Scikit-Learn:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
4. **Documentação Streamlit:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
5. **Materiais de Aula:** Prof. Dr. Márcio Andrey Teixeira, IFSP Campus Catanduva (2026).

---

## 7. Declaração sobre o Uso de IA
Conforme as orientações do professor, informo que contei com o apoio ativo de um assistente de IA (LLM). A ferramenta foi a principal responsável por montar e estruturar o código da aplicação interativa (`app.py` em Streamlit), organizar os testes obrigatórios de validação e formatar o relatório. Por outro lado, o desenvolvimento do script de treinamento (`treinamento.py`) foi realizado em conjunto e acompanhado bloco a bloco por mim, onde estudei, adaptei para o meu estilo de código e executei cada etapa no terminal, validando todos os resultados e métricas.