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
docs/              documentação (Model Card)
models/            artefatos do modelo baseline e campeão
notebooks/         EDA, limpeza dos dados e treinamento do baseline
src/api/routes/    rotas da API
src/core/          configurações da aplicação
src/ml/            modelo e preditor
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
usadas no Colab `(scikit-learn 1.8.0, pandas 2.2.3 e numpy 2.2.0).`

---

## Alternativa com venv (Windows/Linux/Mac)

Criar e ativar o ambiente virtual:

```bash
python -m venv venv
```

```bash
.\venv\Scripts\activate
```

No Linux ou macOS:

```bash
source venv/bin/activate
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

### Dependências de desenvolvimento (para testes)

```bash
pip install -r requirements-dev.txt
```

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
ou com venv:

```bash
python main.py
```

A documentação interativa fica em `/docs` (Swagger) e `/redoc`.

**Endpoints:**

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/health` | Verifica se a API está no ar |
| `POST` | `/predict` | Recebe as 19 features do cliente e retorna a probabilidade de churn |
| `GET` | `/docs` | Documentação Swagger interativa |

**Exemplo de requisição**

```bash
curl -X POST http://localhost:8075/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 1,
    "monthly_charges": 29.85,
    "total_charges": 29.85,
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "No",
    "multiple_lines": "No phone service",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check"
  }'
```

**Resposta esperada**

```json
{
  "churn_probability": 0.8677,
  "churn_prediction": true,
  "threshold": 0.5,
  "model_name": "champion_model",
  "model_version": "0.1.0"
}
```

---

## Deploy

A API está disponível publicamente em: http://177.71.245.99:8075

Acesse a documentação interativa em: http://177.71.245.99:8075/docs

**Endpoints disponíveis:**
- `GET /health`    status da API
- `POST /predict`  predição de churn
- `GET /docs`      documentação Swagger interativa

### Exemplo de requisição

```bash
curl -X POST http://177.71.245.99:8075/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months":1,"monthly_charges":29.85,"total_charges":29.85,"gender":"Female","senior_citizen":"No","partner":"Yes","dependents":"No","phone_service":"No","multiple_lines":"No phone service","internet_service":"DSL","online_security":"No","online_backup":"Yes","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check"}'
```

**Resposta esperada**

```json
{
  "churn_probability": 0.8677,
  "churn_prediction": true,
  "threshold": 0.5,
  "model_name": "champion_model",
  "model_version": "0.1.0"
}
```

---

## Testes

Os testes da API estão em tests/, escritos com pytest e o TestClient do FastAPI.

```bash
uv run pytest
```
ou com venv:

```bash
pytest -v
```

**Cobertura atual:**

- `tests/test_api_health.py` — contrato do `GET /health` e metadados da aplicação.
- `tests/test_api_predict.py` — contrato das 19 features, validação de payload e schemas publicados no OpenAPI.
- `tests/test_preprocessing.py` — funções de limpeza: nomes de coluna, criação do alvo, conversão de `Total Charges` e remoção das colunas de vazamento.
- `tests/test_data.py` — escolha e leitura do arquivo do dataset, com cache local.
- `tests/test_pipelines.py` — construção do pré-processador e dos modelos candidatos.
- `tests/test_evaluation.py` — cálculo das métricas, validação cruzada e escolha do campeão.
- `tests/test_train.py` — orquestração do treinamento e gravação dos artefatos.

**Resultado: 57 testes passando.**

Os testes de pré-processamento e de API usam dados sintéticos, então rodam sem precisar do
dataset do Kaggle nem de credenciais.

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

A comparação completa entre Regressão Logística, Random Forest e MLPClassifier, a escolha do modelo campeão e a matriz de confusão estão documentadas no [Model Card](docs/model_card.md).

---

## Model Card

O Model Card completo está disponível em docs/model_card.md.

---

## Tecnologias Utilizadas

- **Python 3.11**
- **FastAPI** — API REST
- **Scikit-Learn** — modelagem e pré-processamento
- **Pandas / NumPy** — manipulação de dados
- **Pytest** — testes automatizados
- **Docker** — containerização
- **Uvicorn** — servidor ASGI

---

## Licença

Este projeto foi desenvolvido como parte do Tech Challenge — Fase 1 da pós-graduação em Machine Learning Engineering da FIAP.
