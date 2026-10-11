# Credit Default Prediction

Machine learning project for predicting loan default probability from anonymized credit history data.

The project is based on the **Alfa-Bank / ODS Credit Scoring Competition 2022** dataset.

Dataset and competition description:  
https://ods.ai/competitions/dl-fintech-bki/data

## Project Goal

The objective is to predict the probability that a client will default on a loan.

The binary target variable is:

- `flag = 0` — no default
- `flag = 1` — default

The primary evaluation metric is **ROC AUC**.

The project target was to achieve a ROC AUC of at least **0.75** while using a time-aware validation strategy to reduce the risk of overly optimistic evaluation.

## Dataset

The source data contains anonymized credit-history records.

Each application is identified by `id`, while `rn` represents the chronological order of credit products within the client's history.

According to the dataset description:

- larger `id` values correspond to later applications;
- larger `rn` values correspond to later credit products within a client's history;
- many original numerical and categorical variables are anonymized using randomly assigned codes.

Because the encoded values do not necessarily preserve numerical order or distance, they are treated carefully during feature engineering and modeling.

The full dataset contains approximately **3 million applications**.

## Project Structure

```text
credit-default-prediction/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── results/
│   ├── model_results.csv
│   └── temporal_validation_results.csv
├── .gitignore
└── README.md
```

## 1. Exploratory Data Analysis

The exploratory analysis focuses on:

- target class distribution;
- number of credit records per client;
- consistency of credit-history ordering;
- missing values;
- distribution of anonymized categorical variables;
- delinquency indicators;
- differences in credit behavior between default and non-default clients.

The target is highly imbalanced, with defaults representing approximately **3.5%** of applications.

Clients who default tend to show a higher prevalence of delinquency across multiple duration ranges.

## 2. Feature Engineering

Credit-level records are aggregated into a single application-level feature matrix.

The final feature set includes:

- number of credit records;
- delinquency occurrence indicators;
- delinquency shares across credit history;
- first and most recent values of chronological credit attributes;
- categorical frequency distributions;
- distributions of anonymized `pre_*` variables;
- payment-status distributions;
- account and credit-type distributions;
- planned and actual closure indicators;
- recent credit-history features.

Recent behavior is represented using the latest **3 and 5 credit records**.

These features summarize recent:

- delinquency;
- utilization;
- over-limit behavior;
- planned closing indicators;
- actual closing indicators.

Only variables with explicit binary semantics are used for these recent-history calculations to avoid imposing artificial ordering on anonymized category codes.

The final feature matrix contains:

- **3,000,000 applications**
- **547 predictive features**
- **1 target column**

## 3. Validation Strategy

Since application IDs preserve chronological ordering, random train-test splitting is avoided.

The main validation strategy uses:

- first **80%** of applications for training;
- latest **20%** for out-of-time validation.

This simulates a realistic setting where a model trained on historical applications is used to score later applications.

An additional temporal robustness check is also performed:

- first **70%** of applications for training;
- following **10%** for secondary out-of-time validation.

No additional feature engineering or hyperparameter tuning is performed using this secondary validation period.

## 4. Models

Three modeling approaches are compared.

### Logistic Regression

Logistic Regression is used as a linear baseline.

Categorical variables are one-hot encoded and numerical variables are standardized.

### Random Forest

Random Forest is used as a non-linear tree-based alternative capable of capturing feature interactions.

Categorical variables are one-hot encoded, while numerical variables are passed without scaling.

### CatBoost

CatBoost is used as the final gradient boosting model.

It can process categorical features directly, which is especially useful for anonymized encoded variables whose numerical values should not be interpreted as ordered quantities.

Several configurations were evaluated during model development. Intermediate tuning experiments are omitted from the final notebook to keep the modeling workflow concise.

Final configuration:

```text
iterations = 2000
learning_rate = 0.08
depth = 7
l2_leaf_reg = 7
```

Early stopping is used based on validation ROC AUC.

## Results

Performance on the main out-of-time validation period:

| Model | ROC AUC |
|---|---:|
| CatBoost | **0.750233** |
| Logistic Regression | 0.726678 |
| Random Forest | 0.707029 |

CatBoost achieved the best performance and exceeded the project target of **ROC AUC = 0.75**.

Detailed results are available in:

```text
results/model_results.csv
```

## Temporal Robustness

The final CatBoost configuration was additionally evaluated on an earlier chronological validation period.

| Evaluation | Training Period | Validation Period | ROC AUC |
|---|---|---|---:|
| Main out-of-time validation | First 80% | Last 20% | **0.750233** |
| Secondary temporal validation | First 70% | 70–80% | **0.765399** |

The model therefore achieved ROC AUC above **0.75** on two different temporal validation periods.

Performance is higher on the earlier validation period, while the most recent applications are more difficult to predict. This suggests that some temporal distribution shift may be present and highlights the importance of time-aware validation in credit-risk modeling.

Detailed results are available in:

```text
results/temporal_validation_results.csv
```

## Key Findings

- CatBoost provides the strongest performance among the evaluated models.
- Linear Logistic Regression remains a reasonably strong baseline.
- Random Forest performs worse than Logistic Regression for the current feature representation.
- Recent credit behavior provides useful information in addition to full-history aggregates.
- Time-aware validation reveals differences in predictive performance across chronological periods.
- The final model exceeds the target ROC AUC of **0.75** on both evaluated temporal validation periods.

## Reproducing the Project

Run the notebooks in the following order:

```text
01_data_exploration.ipynb
02_feature_engineering.ipynb
03_modeling.ipynb
```

### 1. Download the source data

Download the competition dataset from:

https://ods.ai/competitions/dl-fintech-bki/data

Place the raw data inside the local `data/` directory.

### 2. Run exploratory analysis

```text
notebooks/01_data_exploration.ipynb
```

### 3. Create the feature matrix

```text
notebooks/02_feature_engineering.ipynb
```

The notebook creates:

```text
data/processed/train_features.parquet
```

### 4. Train and evaluate the models

```text
notebooks/03_modeling.ipynb
```

The final CatBoost model is saved locally as:

```text
models/credit_default_catboost.cbm
```

## Repository Data Policy

Large source and generated files are intentionally excluded from the repository.

The following artifacts are not stored on GitHub:

```text
data/
*.parquet
models/
*.cbm
*.cbsnapshot
```

The notebooks contain the complete workflow required to reproduce the feature matrix, train the models, and reproduce the reported evaluation results.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Matplotlib
- Jupyter Notebook
- Parquet

## Final Result

**Best model:** CatBoost  
**Main out-of-time ROC AUC:** **0.750233**  
**Secondary temporal ROC AUC:** **0.765399**
