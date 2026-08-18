# Churn Prediction — Tech Challenge

---

## Descrição do Projeto

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. Este projeto
constrói um pipeline preditivo que classifica clientes com risco de cancelamento, para que
o time de retenção consiga agir de forma preventiva em vez de reativa.

É um problema de classificação binária com dados tabulares, onde cada linha representa um
cliente. O objetivo do projeto é cobrir a análise exploratória, o treinamento e a
disponibilização do modelo por uma API REST.

---

## Estrutura do Repositório

```
data/              arquivos de dados locais (não versionados)
docs/              documentação planejada para as próximas etapas
models/            artefato do modelo baseline
notebooks/         EDA, limpeza dos dados e treinamento do baseline
src/api/routes/    rotas da API
src/core/          configurações da aplicação
src/schemas/       contratos de entrada e saída
tests/             testes automatizados da API
main.py            ponto de entrada da aplicação
```

---

## Setup

O baseline utiliza o dataset público
[Telco Customer Churn — IBM](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset),
disponibilizado no Kaggle.

O projeto usa Python 3.11 e [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente e
as dependências:

```bash
uv sync
```

As versões ficam fixadas no `pyproject.toml` e no `uv.lock`, incluindo as mesmas versões
usadas no Colab (`scikit-learn 1.6.1`, `pandas 2.2.2` e `numpy 2.0.2`).

**Treinamento no Colab:**

Os notebooks baixam o dataset pela API do Kaggle. É preciso cadastrar as credenciais nos
Secrets do Colab (ícone de chave na barra lateral) com os nomes `KAGGLE_USERNAME` e
`KAGGLE_KEY`, obtidos em Settings > API > Create New Token no Kaggle.

---

## API

A estrutura inicial da API foi construída com FastAPI. O código fica em `src/api`
(rotas), `src/core` (configuração, CORS e logging) e `src/schemas` (contratos de
entrada e saída). O ponto de entrada é o `main.py`.

Para subir localmente:

```bash
uv run python main.py
```

A documentação interativa fica em `/docs` (Swagger) e `/redoc`.

**Endpoints:**

- `GET /health` — verifica se a API está no ar.
- `POST /predict` — possui os contratos de entrada e saída definidos e publicados
  no OpenAPI; a inferência ainda não está implementada.

A API ainda não carrega artefatos nem possui serviço de predição.

---

## Testes

Os testes da API estão em `tests/`, escritos com `pytest` e o `TestClient` do
FastAPI. A configuração fica no `pyproject.toml`.

```bash
uv run pytest
```

**Cobertura atual:**

- `tests/test_api_health.py` — contrato do `GET /health` e metadados da aplicação.
- `tests/test_api_predict.py` — contrato das 19 features, validação de payload e
  schemas publicados no OpenAPI.

Os testes cobrem somente a estrutura implementada e não dependem dos artefatos da
Etapa 2.

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

Planejado para a Etapa 4.
