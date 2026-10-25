# Projeto: Tila Health Triage - Classificação de Risco Clínico
# Integrantes: Ryan Aguiar e Pedro Henrique Oliveira Pereira
# Disciplina: Tópicos Avançados em IA - IFSP Catanduva (Prof. Dr. Márcio Andrey Teixeira)

#Importando libs e lendo o dataset obtido no Kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42;
np.random.seed(SEED);

print("Carregando DataSet Kaggle...");

urlDataSet = "https://raw.githubusercontent.com/rishabhpatel9/Healthcare-Triage-Assistant/main/data/raw/synthetic_medical_triage.csv";

#lendo por um dataframe pandas
dfBruto = pd.read_csv(urlDataSet);

#Salvando uma cópia local
dfBruto.to_csv("dataset/triagem_kaggle_bruto.csv", index=False);

print("Dataset carregado com sucesso!!!\n");
print(f"Dimensões originais: {dfBruto.shape[0]} linhas e {dfBruto.shape[1]} colunas.\n");

#Primeiras 5 linhas do dataset
print("Primeiras 5 linhas do dataset:\n");
print(dfBruto.head());

#-----------------------
#Tratamento dos dados
#------------------------

print("\nVerificando tipos de dados:\n");
print(dfBruto.dtypes);

#Verificando percentual de valores nulos
print("\nPercentual de valores nulos:\n");
print((dfBruto.isna().mean() * 100).round(2));

dfTratado = dfBruto.copy();

#Tratamento da variável "arrival_mode"
#walk_in = 0 | wheelchair = 1 | ambulance = 2
dfTratado["arrival_mode"] = dfTratado["arrival_mode"].map({"walk_in": 0, "wheelchair": 1, "ambulance": 2}).astype(int);

#Mapeando o alvo para 3 classes de triagem hospitalar (Baixa, Média e Alta)
def agruparPrioridade(nivel):
    if nivel == 0:
        return 0; #Baixa Prioridade (Verde)
    elif nivel in [1, 2]:
        return 1; #Média Prioridade (Amarelo)
    else:
        return 2; #Alta Prioridade (Vermelho / Emergência)

dfTratado["prioridade"] = dfTratado["triage_level"].apply(agruparPrioridade);

dfTratado = dfTratado.drop(columns=["triage_level"]);

print("\nDistribuição da Nova Variável Alvo (prioridade):");
print(dfTratado["prioridade"].value_counts().sort_index());
print("\nPercentual (%):");
print((dfTratado["prioridade"].value_counts(normalize=True).sort_index() * 100).round(2));

print("\nResumo Estatístico das Variáveis Numéricas:");
print(dfTratado.describe().round(2).T[["mean", "std", "min", "50%", "max"]]);


#-----------------------
#Separação x / y, divisão entre treino e teste e normalização dos dados
#------------------------
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;

#Lista das features
nomesFeatures =[
    "age", "heart_rate", "systolic_blood_pressure", 
    "oxygen_saturation", "body_temperature", "pain_level", 
    "chronic_disease_count", "previous_er_visits", "arrival_mode"
];

nomesClasses =[
    "Baixa (Verde)", "Média (Amarelo)", "Alta (Vermelho)"
];

#Separando a matriz de features X e o vetor alvo y
x = dfTratado[nomesFeatures];
y = dfTratado["prioridade"];

#Divisão do dataset em 75% trino e 25% teste
xTreino, xTeste, yTreino, yTeste = train_test_split(
    x, y, test_size=0.25, random_state=SEED, stratify=y
);

#stratify=y faz com que mesmo que as proporções disproporcionais do dataset sejam preservadas em ambos os conjuntos.

print("\nDivisão Treino e Teste:");
print(f"xTreino (Treinamento): {xTreino.shape[0]} amostras e {xTreino.shape[1]} atributos.");
print(f"xTeste  (Teste): {xTeste.shape[0]} amostras e {xTeste.shape[1]} atributos.");
print("\nDistribuição de classes no conjunto de Teste:");
print(yTeste.value_counts().sort_index());


#Feature Scaling (Padronização)
scaler = StandardScaler();
xTreinoScaled = scaler.fit_transform(xTreino);
xTesteScaled = scaler.transform(xTeste);

print("\nPadronização com StandardScaler aplicada.");
print(f"Média pós-escalonamento no treino: {xTreinoScaled.mean():.4f} (próxima de 0)");
print(f"Desvio padrão pós-escalonamento no treino: {xTreinoScaled.std():.4f} (igual a 1)");

#-----------------------
#Modelo 01 - Regressão logística - treinamento
#------------------------
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
);
import os;

os.makedirs("figuras", exist_ok=True);

#Instanciando a Regressão Logística com mil iterações (convergência garantida)
modeloLR = LogisticRegression(max_iter=1000, random_state=SEED);

#Treinando o modelo com os dados escalados
modeloLR.fit(xTreinoScaled, yTreino);

#Predição no conjunto de teste
yPredLR = modeloLR.predict(xTesteScaled);
yProbLR = modeloLR.predict_proba(xTesteScaled);

#Métricas
acuraciaLR = accuracy_score(yTeste, yPredLR);
f1MacroLR = f1_score(yTeste, yPredLR, average='macro');
recallEmergenciaLR = recall_score(yTeste, yPredLR, average=None)[2]; #[2] Alta/Emergência

print(f"\nAcurácia global (Regressão Logística): {acuraciaLR:.4f} ({acuraciaLR * 100:.2f}%)");
print(f"F1-Score macro (Equilíbrio das classes): {f1MacroLR:.4f}");
print(f"Recall da classe crítica (Emergência): {recallEmergenciaLR:.4f} ({recallEmergenciaLR * 100:.2f}%)\n");

print("Relatório de Classificação");
print(classification_report(yTeste, yPredLR, target_names=nomesClasses, digits=4));

#Matriz de confusão
matrizConfusaoLR = confusion_matrix(yTeste, yPredLR);
print("Matriz de confusão:");
print(matrizConfusaoLR);

#Gerando e salvando o gráfico da matriz de confusão
fig, ax = plt.subplots(figsize=(6, 5));
dispLR = ConfusionMatrixDisplay(confusion_matrix=matrizConfusaoLR, display_labels=nomesClasses);
dispLR.plot(ax=ax, cmap="Blues", values_format="d");
plt.title("Matriz de Confusão - Regressão Logística", fontsize=11, fontweight="bold");
plt.tight_layout();
plt.savefig("figuras/matriz_confusao_regressao_logistica.png", dpi=300);
plt.close();

#-----------------------
#Modelo 02 - Árvore de decisão - treinamento
#------------------------
from sklearn.tree import DecisionTreeClassifier;

#Instanciando a árvore de decisão com controle de profundidade para evitar overfitting (treinar demais o modelo)
modeloDT = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=6,
    min_samples_leaf=10,
    random_state=SEED
);

#Treinamento da Árvore
modeloDT.fit(xTreino, yTreino);

#Predição no conjunto de teste
yPredDT = modeloDT.predict(xTeste);
yProbDT = modeloDT.predict_proba(xTeste);

#Métricas
acuraciaDT = accuracy_score(yTeste, yPredDT);
f1MacroDT = f1_score(yTeste, yPredDT, average="macro");
recallEmergenciaDT = recall_score(yTeste, yPredDT, average=None)[2];

print(f"\nAcurácia global (Árvore de Decisão): {acuraciaDT:.4f} ({acuraciaDT * 100:.2f}%)");
print(f"F1-Score macro (Equilíbrio das classes): {f1MacroDT:.4f}");
print(f"Recall da classe crítica (Emergência): {recallEmergenciaDT:.4f} ({recallEmergenciaDT * 100:.2f}%)\n");


print("Relatório de classificação:");
print(classification_report(yTeste, yPredDT, target_names=nomesClasses, digits=4));

#Matriz de Confusão da Árvore
matrizConfusaoDT = confusion_matrix(yTeste, yPredDT);
print("Matriz de confusão:");
print(matrizConfusaoDT);

#Gráfico da matriz de confusão
fig, ax = plt.subplots(figsize=(6, 5));
dispDT = ConfusionMatrixDisplay(confusion_matrix=matrizConfusaoDT, display_labels=nomesClasses);
dispDT.plot(ax=ax, cmap="Greens", values_format="d");
plt.title("Matriz de Confusão - Árvore de Decisão", fontsize=11, fontweight="bold");
plt.tight_layout();
plt.savefig("figuras/matriz_confusao_arvore_decisao.png", dpi=300);
plt.close();

#extração das features para descobrir quais sinais vitais foram mais importantes para a tomada de decisão da árvore de decisão
print("\nImportância das features");
dfImportancias = pd.DataFrame({
    "Atributo": nomesFeatures,
    "Importancia": modeloDT.feature_importances_
}).sort_values(by="Importancia", ascending=False);

for _, linha in dfImportancias.iterrows():
    print(f"{linha['Atributo']:<26}: {linha['Importancia'] * 100:>6.2f}%");


#Gráfico de barras da importância dos atributos
plt.figure(figsize=(8, 4.5));
sns.barplot(data=dfImportancias, x='Importancia', y='Atributo', color='#2e7d32');
plt.title("Relevância dos Atributos na Decisão Clínica (Árvore de Decisão)", fontsize=11, fontweight="bold");
plt.xlabel("Importância Relativa (Ganho de Informação / Entropia)");
plt.ylabel("Atributo Clínico");
plt.tight_layout();
plt.savefig("figuras/importancia_atributos_dt.png", dpi=300);
plt.close();

#-----------------------
#Comparação entre os modelos
#------------------------

#Criando tabela comparativa
tabelaComparativa = pd.DataFrame({
    'Modelo': ['Regressão Logística', 'Árvore de Decisão'],
    'Acurácia Global': [acuraciaLR, acuraciaDT],
    'F1-Score Macro': [f1MacroLR, f1MacroDT],
    'Recall (Emergência)': [recallEmergenciaLR, recallEmergenciaDT]
});

print(tabelaComparativa.round(4).to_string(index=False));

#Gerando gráfico de barras comparando os dois modelos
plt.figure(figsize=(8, 5));
posicoes = np.arange(len(tabelaComparativa));
largura = 0.25;

plt.bar(posicoes - largura, tabelaComparativa['Acurácia Global'], width=largura, label='Acurácia Global', color='#1976d2');
plt.bar(posicoes, tabelaComparativa['F1-Score Macro'], width=largura, label='F1-Score Macro', color='#388e3c');
plt.bar(posicoes + largura, tabelaComparativa['Recall (Emergência)'], width=largura, label='Recall (Emergência)', color='#d32f2f');

plt.xticks(posicoes, tabelaComparativa['Modelo'], fontweight='bold');
plt.ylim(0, 1.15);
plt.ylabel("Pontuação (0.0 a 1.0)");
plt.title("Comparativo de desempenho: Regressão logística vs Árvore de decisão", fontsize=11, fontweight="bold");
plt.legend(loc='lower right');
plt.tight_layout();
plt.savefig("figuras/comparacao_modelos.png", dpi=300);
plt.close();

#Definindo o modelo vencedor para o Deploy
if recallEmergenciaLR >= recallEmergenciaDT:
    modeloVencedor = modeloLR;
    nomeModeloVencedor = "Regressão Logística";
    precisaScaler = True;
    metricaVencedor = {
        'acc': acuraciaLR,
        'f1_macro': f1MacroLR,
        'recall_emergencia': recallEmergenciaLR
    };
else:
    modeloVencedor = modeloDT;
    nomeModeloVencedor = "Árvore de Decisão";
    precisaScaler = False;
    metricaVencedor = {
        'acc': acuraciaDT,
        'f1_macro': f1MacroDT,
        'recall_emergencia': recallEmergenciaDT
    };

print(f"\nModelo escolhido para o deploy: {nomeModeloVencedor}");
print("Justificativa: maior sensibilidade (Recall) na detecção de emergências críticas e melhor equilíbrio macro.");


#-----------------------
#Serialização do Modelo
#-----------------------
import pickle;

os.makedirs("modelo", exist_ok=True);

caminhoModelo = "modelo/triagem_model.pkl";

dadosExportacao = {
    'modelo': modeloVencedor,
    'scaler': scaler,
    'precisa_scaler': precisaScaler,
    'tipo_modelo': nomeModeloVencedor,
    'feature_names': nomesFeatures,
    'target_names': nomesClasses,
    'metricas': metricaVencedor
};

#Salvando no disco
with open(caminhoModelo, 'wb') as arquivoModelo:
    pickle.dump(dadosExportacao, arquivoModelo);