"""
Aplicação Streamlit: Tila Health Triage
Classificação de Risco Clínico no Acolhimento
Disciplina: Tópicos Avançados em Inteligência Artificial - IFSP Campus Catanduva
Professor: Prof. Dr. Márcio Andrey Teixeira
Aluno: Ryan Aguiar
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Tila Health Triage - Classificação de Risco",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregando o modelo e o scaler em cache
@st.cache_resource
def carregarModelo():
    caminho = "modelo/triagem_model.pkl"
    if not os.path.exists(caminho):
        caminho = os.path.join(os.path.dirname(__file__), "modelo", "triagem_model.pkl")
    with open(caminho, "rb") as f:
        dados = pickle.load(f)
    return dados

try:
    dadosArtefato = carregarModelo()
    modelo = dadosArtefato['modelo']
    scaler = dadosArtefato['scaler']
    precisaScaler = dadosArtefato['precisa_scaler']
    tipoModelo = dadosArtefato['tipo_modelo']
    nomesFeatures = dadosArtefato['feature_names']
    nomesClasses = dadosArtefato['target_names']
    metricas = dadosArtefato.get('metricas', {})
except Exception as e:
    st.error(f"Erro ao carregar o arquivo do modelo: {e}")
    st.info("Execute 'python treinamento.py' para gerar o arquivo do modelo antes de iniciar o app.")
    st.stop()

# Sidebar com informações do projeto
st.sidebar.title("🏥 Tila Health Triage")
st.sidebar.markdown("""
**IFSP - Campus Catanduva**  
*Tópicos Avançados em IA*  
**Professor:** Prof. Dr. Márcio Andrey Teixeira  
**Alunos:** Ryan Aguiar e Pedro Henrique Oliveira Pereira  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Informações do Modelo")
st.sidebar.markdown(f"**Algoritmo:** `{tipoModelo}`")
st.sidebar.markdown(f"**Acurácia Global:** `{metricas.get('acc', 0.9622) * 100:.2f}%`")
st.sidebar.markdown(f"**Recall (Emergência):** `{metricas.get('recall_emergencia', 0.9327) * 100:.2f}%`")
st.sidebar.markdown(f"**Base de Dados:** `Kaggle Triage Dataset (18.000 registros)`")
st.sidebar.markdown("---")
st.sidebar.caption("Aplicação desenvolvida para o Trabalho Prático 1.")

# Título principal
st.title("Sistema de Triagem e Classificação de Risco Clínico")
st.markdown("""
Esta aplicação foi desenvolvida como parte do trabalho da disciplina de IA para auxiliar a equipe de acolhimento e enfermagem 
na priorização de pacientes que chegam ao pronto atendimento, sugerindo a fila correta de acordo com a gravidade dos sinais vitais.
""")

# Abas da aplicação
abaForm, abaDicionario, abaMetricas = st.tabs([
    "📋 Formulário de Entrada",
    "📖 Dicionário de Campos",
    "📊 Resultados e Métricas"
])

# -------------------------------------------------------------
# ABA 1: FORMULÁRIO DE ENTRADA E PREDIÇÃO
# -------------------------------------------------------------
with abaForm:
    st.subheader("Entrada de Dados do Paciente")
    st.markdown("Preencha os dados aferidos durante o acolhimento do paciente:")

    with st.form("form_triagem"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Perfil & Chegada")
            idade = st.number_input(
                "Idade do Paciente (anos):",
                min_value=0.0,
                max_value=120.0,
                value=42.0,
                step=1.0,
                help="Idade em anos."
            )

            arrival_label = st.selectbox(
                "Modo de Chegada ao Hospital:",
                options=[
                    "0 - A pé / Condução própria (walk_in)",
                    "1 - Cadeira de Rodas (wheelchair)",
                    "2 - Ambulância (ambulance)"
                ],
                index=0,
                help="Como o paciente chegou à unidade de saúde."
            )
            arrival_mode = int(arrival_label.split(" - ")[0])

            pain_level = st.slider(
                "Nível de Dor Autorrelatada (Escala 1 a 10):",
                min_value=1,
                max_value=10,
                value=3,
                help="1 = Dor leve / desconforto mínimo; 10 = Dor extrema."
            )

        with col2:
            st.markdown("##### 💓 Sinais Vitais Cardiovasculares")
            systolic_bp = st.number_input(
                "Pressão Arterial Sistólica - PAS (mmHg):",
                min_value=40.0,
                max_value=260.0,
                value=120.0,
                step=1.0,
                help="Pressão arterial máxima aferida."
            )

            heart_rate = st.number_input(
                "Frequência Cardíaca - FC (BPM):",
                min_value=30.0,
                max_value=220.0,
                value=78.0,
                step=1.0,
                help="Batimentos por minuto."
            )

            body_temp = st.number_input(
                "Temperatura Corporal (°C):",
                min_value=32.0,
                max_value=43.0,
                value=36.8,
                step=0.1,
                format="%.2f",
                help="Temperatura axilar em graus Celsius."
            )

        with col3:
            st.markdown("##### 🫁 Oxigenação e Histórico")
            oxygen_sat = st.number_input(
                "Saturação de Oxigênio - SpO₂ (%):",
                min_value=50.0,
                max_value=100.0,
                value=98.0,
                step=0.5,
                help="Saturação periférica de O₂ por oximetria."
            )

            chronic_diseases = st.number_input(
                "Número de Doenças Crônicas:",
                min_value=0,
                max_value=15,
                value=1,
                step=1,
                help="Quantidade de comorbidades prévias (hipertensão, diabetes, asma, etc.)."
            )

            previous_visits = st.number_input(
                "Visitas Anteriores ao Pronto-Socorro:",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Histórico de idas anteriores ao pronto atendimento."
            )

        st.markdown("")
        btnPrever = st.form_submit_button("🩺 Realizar Predição de Risco", type="primary", use_container_width=True)

    if btnPrever:
        errosValidacao = []

        if systolic_bp < 60 or systolic_bp > 240:
            errosValidacao.append(f"Pressão Sistólica de {systolic_bp} mmHg está fora dos limites fisiológicos aceitáveis.")
        if heart_rate < 35 or heart_rate > 200:
            errosValidacao.append(f"Frequência Cardíaca de {heart_rate} BPM é extrema ou incoerente.")
        if body_temp < 34.0 or body_temp > 42.5:
            errosValidacao.append(f"Temperatura de {body_temp} °C é biologicamente incompatível.")
        if oxygen_sat < 65:
            errosValidacao.append(f"Saturação de {oxygen_sat}% é gravíssima; verifique o posicionamento do oxímetro.")

        if errosValidacao:
            st.error("⚠️ **Valores inválidos ou suspeitos identificados:**")
            for err in errosValidacao:
                st.write(f"- {err}")
            st.warning("Verifique os campos informados antes de prosseguir com a inferência.")
        else:
            entradaDados = pd.DataFrame([{
                'age': idade,
                'heart_rate': heart_rate,
                'systolic_blood_pressure': systolic_bp,
                'oxygen_saturation': oxygen_sat,
                'body_temperature': body_temp,
                'pain_level': pain_level,
                'chronic_disease_count': chronic_diseases,
                'previous_er_visits': previous_visits,
                'arrival_mode': arrival_mode
            }])

            if precisaScaler:
                entradaFinal = scaler.transform(entradaDados)
            else:
                entradaFinal = entradaDados.values

            predicao = modelo.predict(entradaFinal)[0]
            probabilidades = modelo.predict_proba(entradaFinal)[0]

            st.markdown("---")
            st.markdown("### 🎯 Resultado da Classificação")

            colRes1, colRes2 = st.columns([1.2, 1])

            with colRes1:
                if predicao == 0:
                    st.success("""
                    ### 🟢 BAIXA PRIORIDADE (VERDE)
                    **Classificação:** Pouco Urgente  
                    **Tempo máximo sugerido:** Até 120 minutos  
                    **Conduta:** Paciente clinicamente estável, sem sinais de gravidade aguda. Pode aguardar atendimento na sala de espera normal.
                    """)
                elif predicao == 1:
                    st.warning("""
                    ### 🟡 MÉDIA PRIORIDADE (AMARELO)
                    **Classificação:** Urgência Relativa  
                    **Tempo máximo sugerido:** Até 60 minutos  
                    **Conduta:** Paciente com queixas moderadas ou alteração nos sinais vitais. Encaminhar para observação e reavaliação periódica.
                    """)
                else:
                    st.error("""
                    ### 🔴 ALTA PRIORIDADE (VERMELHO)
                    **Classificação:** EMERGÊNCIA IMEDIATA  
                    **Tempo máximo sugerido:** ZERO MINUTOS (Atendimento Imediato)  
                    **Conduta:** Encaminhar imediatamente para a Sala de Emergência/Estabilização. Acionar equipe médica e suporte de urgência.
                    """)

            with colRes2:
                st.markdown("##### Probabilidades Estimadas pelo Modelo:")
                st.write(f"🟢 **Baixa (Verde):** `{probabilidades[0] * 100:.1f}%`")
                st.progress(float(probabilidades[0]))

                st.write(f"🟡 **Média (Amarelo):** `{probabilidades[1] * 100:.1f}%`")
                st.progress(float(probabilidades[1]))

                st.write(f"🔴 **Alta (Vermelho):** `{probabilidades[2] * 100:.1f}%`")
                st.progress(float(probabilidades[2]))

# -------------------------------------------------------------
# ABA 2: DICIONÁRIO DE CAMPOS
# -------------------------------------------------------------
with abaDicionario:
    st.subheader("Dicionário de Atributos e Unidades")
    st.markdown("Detalhamento das variáveis clínicas utilizadas pelo modelo:")

    tabelaDicionario = [
        {"Campo": "age", "Tipo": "Numérico (Float)", "Unidade": "Anos", "Valores Típicos": "0 a 95 anos", "Descrição": "Idade cronológica do paciente."},
        {"Campo": "heart_rate", "Tipo": "Numérico (Float)", "Unidade": "BPM", "Valores Típicos": "60 a 100 BPM", "Descrição": "Frequência de batimentos cardíacos por minuto."},
        {"Campo": "systolic_blood_pressure", "Tipo": "Numérico (Float)", "Unidade": "mmHg", "Valores Típicos": "90 a 140 mmHg", "Descrição": "Pressão arterial sistólica máxima."},
        {"Campo": "oxygen_saturation", "Tipo": "Numérico (Float)", "Unidade": "%", "Valores Típicos": "95 a 100%", "Descrição": "Saturação de oxigênio no sangue (SpO₂)."},
        {"Campo": "body_temperature", "Tipo": "Numérico (Float)", "Unidade": "°C", "Valores Típicos": "36.0 a 37.5 °C", "Descrição": "Temperatura axilar corporal."},
        {"Campo": "pain_level", "Tipo": "Numérico (Int)", "Unidade": "Escala 1-10", "Valores Típicos": "1 a 10", "Descrição": "Intensidade da dor autorrelatada pelo paciente."},
        {"Campo": "chronic_disease_count", "Tipo": "Numérico (Int)", "Unidade": "Quantidade", "Valores Típicos": "0 a 10", "Descrição": "Número de doenças crônicas diagnosticadas previamente."},
        {"Campo": "previous_er_visits", "Tipo": "Numérico (Int)", "Unidade": "Quantidade", "Valores Típicos": "0 a 11", "Descrição": "Quantidade de idas anteriores ao pronto-socorro."},
        {"Campo": "arrival_mode", "Tipo": "Categórico (Int)", "Unidade": "Código", "Valores Típicos": "0, 1 ou 2", "Descrição": "0 = a pé/carro próprio, 1 = cadeira de rodas, 2 = ambulância."}
    ]
    st.dataframe(pd.DataFrame(tabelaDicionario), use_container_width=True)

# -------------------------------------------------------------
# ABA 3: RESULTADOS E MÉTRICAS
# -------------------------------------------------------------
with abaMetricas:
    st.subheader("Comparação dos Modelos Testados")
    st.markdown("""
    Durante o trabalho, comparamos duas abordagens clássicas ensinadas em aula: **Regressão Logística** e **Árvore de Decisão**, 
    avaliadas em 4.500 amostras inéditas do conjunto de teste (de um total de 18.000 pacientes reais).
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia Global no Teste", "96,22%", "+2,04% vs Árvore")
    c2.metric("F1-Score Macro", "0,9555", "Equilíbrio entre classes")
    c3.metric("Recall da Emergência (Vermelho)", "93,27%", "Sensibilidade em casos graves")

    st.markdown("---")
    st.markdown("##### Matriz de Confusão no Teste (4.500 pacientes)")
    
    dfMatriz = pd.DataFrame(
        [[2422, 59, 0],
         [86, 1700, 10],
         [0, 15, 208]],
        columns=["Previsto: Baixa (Verde)", "Previsto: Média (Amarelo)", "Previsto: Alta (Vermelho)"],
        index=["Real: Baixa (Verde)", "Real: Média (Amarelo)", "Real: Alta (Vermelho)"]
    )
    st.table(dfMatriz)

    st.info("""
    **Destaque Importante:** Nenhum paciente real de Alta Prioridade (Emergência) foi erroneamente classificado como Baixa Prioridade (célula igual a 0). 
    Os poucos casos de discordância (15 pacientes) foram colocados como Média Prioridade, mantendo-os em observação médica.
    """)

st.markdown("---")
st.caption("Desenvolvido por Ryan Aguiar e Pedro Henrique Oliveira Pereira — Disciplina de Tópicos Avançados em IA — IFSP Catanduva (2026)")