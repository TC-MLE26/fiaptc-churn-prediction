# Model Card — Predição de Churn

## Identificação

| | |
|---|---|
| **Nome** | Churn Prediction — Telco |
| **Versão** | 1.0.0 |
| **Tipo** | Classificação binária |
| **Algoritmo** | Regressão Logística com `class_weight="balanced"` |
| **Artefato** | `models/champion_model.joblib` |
| **Framework** | scikit-learn 1.8.0 |
| **Seed** | 42 |

**Para que serve:** estimar a probabilidade de um cliente de telecomunicações cancelar o
serviço, para que o time de retenção possa priorizar contatos preventivos.

---

## Dados

**Origem:** [Telco Customer Churn — IBM](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset), dataset público do Kaggle.

- 7.043 clientes, 33 colunas no arquivo original
- Após a limpeza: 19 features (3 numéricas e 16 categóricas) e a variável alvo
- 73,5% permaneceram e 26,5% cancelaram
- Divisão: 80% treino e 20% teste, estratificada, com seed fixa

Foram removidas as colunas que vazam o alvo (`Churn Label`, `Churn Value`, `Churn Score`,
`Churn Reason` e `CLTV`), porque só são preenchidas depois que o cliente cancela e não
existiriam no momento da predição. Também saíram identificadores e colunas geográficas.

A coluna `Total Charges` vinha como texto e foi convertida para numérico. Os 11 registros
em branco são de clientes com zero meses de contrato, que ainda não foram cobrados, e
receberam valor zero.

---

## Performance

**Métrica técnica:** F1 da classe churn. **Métrica de negócio:** recall da classe churn.
Acurácia foi descartada porque, com 26,5% de churn, prever que ninguém cancela já daria
73,5% de acerto sem utilidade nenhuma.

### Conjunto de teste

| Métrica | Valor |
|---|---|
| F1 (churn) | 0,617 |
| Recall (churn) | 0,781 |
| Precisão (churn) | 0,510 |
| AUC-ROC | 0,849 |

### Matriz de confusão (1.409 clientes, 374 com churn real)

| | Previu que fica | Previu churn |
|---|---|---|
| **Ficou (1.035)** | 755 | 280 |
| **Cancelou (374)** | 82 | 292 |

De 374 clientes que iriam cancelar, o modelo identifica 292 e deixa passar 82. Em troca,
gera 280 alarmes falsos: cerca de metade dos clientes sinalizados não iria cancelar.

Isso é uma escolha, não um defeito. Perder um cliente custa mais caro que ligar para quem
não ia cancelar, então o modelo foi ajustado para errar para esse lado.

### Comparação com os outros modelos

| Modelo | F1 | Recall | AUC-ROC | F1 validação cruzada |
|---|---|---|---|---|
| **Regressão Logística (`balanced`)** | **0,617** | **0,781** | 0,849 | **0,640 ± 0,021** |
| Regressão Logística | 0,604 | 0,570 | 0,849 | 0,621 ± 0,029 |
| Random Forest | 0,591 | 0,543 | 0,852 | 0,606 ± 0,026 |
| MLPClassifier | 0,559 | 0,473 | 0,844 | 0,615 ± 0,033 |

Validação cruzada com 5 folds estratificados. O Random Forest teve AUC levemente melhor,
mas perdeu em F1 e recall. Escolhemos a Regressão Logística por ser mais consistente na
validação cruzada, mais simples e mais rápida de servir.

---

## Limitações

- **Só Califórnia.** Todos os clientes do dataset são de uma região dos Estados Unidos.
  Não há garantia de que o padrão se repita em outros lugares.
- **Uma única operadora.** O modelo aprendeu o comportamento da base de uma empresa
  específica.
- **Faltam informações importantes.** Não há dados de uso do serviço, de qualidade da
  conexão nem de reclamações no atendimento — sinais fortes de insatisfação que o modelo
  não enxerga.
- **Metade dos alertas é falso positivo.** Com precisão de 0,510, o time de retenção vai
  contatar bastante gente que não ia cancelar.
- **O limiar de 0,5 é o padrão**, não foi calculado a partir do custo real de contatar um
  cliente.
- **Depende da versão da biblioteca.** O modelo foi salvo com scikit-learn 1.8.0 e pode
  falhar ao ser carregado em outra versão.

---

## Vieses

- **O modelo usa gênero e faixa etária.** As features `gender` e `senior_citizen` entram na
  predição. Isso significa que características pessoais influenciam quem é sinalizado para
  receber uma oferta de retenção. **Não avaliamos o desempenho separado por esses grupos**,
  então não sabemos se o modelo trata todos igualmente.
- **O modelo repete o passado.** Ele aprendeu com cancelamentos que já aconteceram, sob as
  políticas de retenção da época. Se havia alguma falha no atendimento a um grupo de
  clientes, o modelo tende a reproduzi-la.
- **Não deve decidir sozinho.** A saída serve para priorizar contatos. Não deve ser usada
  para negar serviço, piorar atendimento ou cobrar mais de quem tem risco alto.
