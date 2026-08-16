# Churn Prediction — Tech Challenge

---

## Descrição do Projeto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. Este projeto
constrói um pipeline preditivo que classifica clientes com risco de cancelamento, para que
o time de retenção consiga agir de forma preventiva em vez de reativa.

É um problema de classificação binária com dados tabulares, onde cada linha representa um
cliente. O projeto vai da análise exploratória até o modelo servido por uma API REST.

---

## Estrutura do Repositório

```
data/         dados brutos (o dataset é baixado do Kaggle, não versionado)
docs/         documentação e Model Card
models/       modelos treinados (.joblib)
notebooks/    experimentação: EDA, baseline e comparação de modelos
src/          código produtivo (pré-processamento, predição e API)
tests/        testes automatizados com pytest
```

---

## Setup

O modelo foi treinado com base num dataset de casos de churn, open dataset no kaggle: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

**Ambiente local** (usado para servir a API e rodar os testes, não para treinar):

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

As versões no `requirements.txt` estão fixadas nas mesmas usadas no Colab
(`scikit-learn 1.6.1`, `pandas 2.2.2`, `numpy 2.0.2`). Se divergirem, o `joblib.load` do
modelo pode falhar ou emitir aviso de incompatibilidade. Use Python 3.11, que é a versão
compatível com esse conjunto de bibliotecas.

**Treinamento no Colab:**

Os notebooks baixam o dataset pela API do Kaggle. É preciso cadastrar as credenciais nos
Secrets do Colab (ícone de chave na barra lateral) com os nomes `KAGGLE_USERNAME` e
`KAGGLE_KEY`, obtidos em Settings > API > Create New Token no Kaggle.

---

## Como Executar

---

## API

Ainda não implementada. Será construída com FastAPI, com dois endpoints:

- `GET /health` — verifica se a API está no ar
- `POST /predict` — recebe os dados de um cliente e retorna a propensão de churn

---

## Testes

Ainda não implementados (Etapa 3).

---

## Modelo e Resultados

**Dataset:** 7.043 clientes, 33 colunas no arquivo original. Após a limpeza restaram 19
features (3 numéricas e 16 categóricas) e a variável alvo.

**Tratamento aplicado:** foram removidas as colunas que vazam o alvo (`Churn Label`,
`Churn Value`, `Churn Score`, `Churn Reason` e `CLTV`), já que só são preenchidas depois
que o cliente cancela e não existiriam em produção. Também saíram identificadores e
colunas geográficas. A coluna `Total Charges` vinha como texto e foi convertida para
numérico, com 11 registros em branco preenchidos com zero (clientes com zero meses de
contrato, que ainda não foram cobrados).

**Balanceamento:** 73,5% dos clientes permaneceram e 26,5% cancelaram.

**Métricas escolhidas:** F1 da classe churn como métrica técnica e AUC-ROC como secundária.
A métrica de negócio é o recall da classe churn, porque deixar de identificar um cliente
que vai cancelar custa mais caro que contatar um cliente que não iria cancelar. Acurácia
foi descartada: com 26,5% de churn, prever que ninguém cancela já resultaria em 73,5% de
acerto.

**Baseline (Regressão Logística):**

| Métrica | Padrão | Com `class_weight="balanced"` |
|---|---|---|
| F1 (churn) | 0,604 | 0,617 |
| Recall (churn) | 0,570 | 0,781 |
| AUC-ROC | 0,849 | 0,849 |

O AUC idêntico mostra que o modelo balanceado não é mais preciso, apenas desloca o erro
para o lado que interessa ao negócio: identifica 78% de quem iria cancelar, contra 57% da
versão padrão.

A comparação com Random Forest e MLP, e a escolha do modelo campeão, são da Etapa 2.

---

## Model Card

Ainda não escrito (Etapa 4).

---

## Contribuidores

-
