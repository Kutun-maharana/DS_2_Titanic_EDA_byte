# Titanic Dataset Exploratory Data Analysis

## AVIP 2026 Data Science Task 2

This project is a complete, beginner-friendly exploratory data analysis (EDA) of the Kaggle Titanic `train.csv` dataset. It focuses on data quality, transparent cleaning, simple feature engineering, descriptive analysis, and reproducible charts. **No machine-learning model is included or required.**

## Objective

- Inspect the Titanic training data and document its quality.
- Handle missing values without fabricating information.
- Create `FamilySize`, `IsAlone`, `AgeGroup`, and `CabinKnown`.
- Analyze survival in relation to passenger class, gender, age, fare, family size, and embarkation port.
- Communicate at least three data-supported observations while distinguishing association from causation.

## Dataset

The dataset was downloaded from Kaggle:

https://www.kaggle.com/competitions/titanic/data

Only the Kaggle competition training file is used. The project does not require or include `test.csv` or `gender_submission.csv`.

The source file is [data/train.csv](data/train.csv). It contains passenger details and the observed `Survived` outcome (`0` = did not survive, `1` = survived).

## Project structure

```text
DS_2_Titanic_EDA_byte/
├── data/
│   ├── train.csv
│   └── cleaned_train.csv
├── notebooks/
│   └── titanic_eda.ipynb
├── charts/
│   ├── survival_by_class_gender.png
│   ├── age_distribution.png
│   ├── correlation_heatmap.png
│   ├── additional_distributions.png
│   └── age_group_and_family_size.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Data quality and cleaning

The notebook reports the dataset shape, column names, data types, descriptive statistics, useful unique-value counts, missing values, duplicate rows, and data-quality observations before cleaning.

Cleaning decisions:

1. Duplicate rows are checked and removed if present.
2. Missing `Age` values are filled with the median age calculated from the dataset.
3. Missing `Embarked` values are filled with the mode, the most frequent port.
4. `Cabin` is not imputed because most cabin values are missing and inventing cabin identifiers would be misleading.
5. `CabinKnown` is created to indicate whether a cabin was recorded.
6. Before-and-after missing-value counts are displayed.

The final cleaned and feature-engineered dataframe is saved reproducibly as [data/cleaned_train.csv](data/cleaned_train.csv) when the notebook runs.

## Feature engineering

- `FamilySize = SibSp + Parch + 1`: counts the passenger and accompanying family members.
- `IsAlone`: equals `1` when `FamilySize` is one, otherwise `0`.
- `AgeGroup`: groups passengers into child, teenager, young adult, adult, and senior categories.
- `CabinKnown`: equals `1` when a cabin value is present, otherwise `0`.

These features are simple analytical summaries; no unnecessary or model-specific features are added.

## Visualizations

Required charts are exported to the single root-level `charts/` directory:

- `survival_by_class_gender.png`: survival rate by passenger class and gender.
- `age_distribution.png`: passenger age histogram with labels, suitable bins, and grid.
- `correlation_heatmap.png`: correlations among meaningful numerical columns including `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, and `IsAlone`.

Additional useful charts:

- `additional_distributions.png`: overall survival, survival by class, and fare distribution.
- `age_group_and_family_size.png`: survival by age group and family-size analysis.

## Key findings

The notebook calculates the statistics from the dataset rather than hardcoding results. The analysis documents that:

- Observed survival rates vary across passenger classes.
- Observed survival rates differ between female and male passengers, including within passenger classes.
- Age groups and family-size groups provide useful descriptive segmentation.
- Fare and passenger class are related and should be interpreted together.

These are associations in the Titanic training data, not proof that any one variable caused survival outcomes.

## Technologies used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter
- IPython kernel

## Installation

From the project root:

```powershell
python -m pip install -r requirements.txt
```

The required packages are listed in [requirements.txt](requirements.txt).

## How to run

Start Jupyter:

```powershell
jupyter notebook
```

Open [notebooks/titanic_eda.ipynb](notebooks/titanic_eda.ipynb) and run all cells from top to bottom. The notebook resolves the project root so that it reads `data/train.csv`, saves `data/cleaned_train.csv`, and exports all charts to the root `charts/` directory.

For non-interactive validation:

```powershell
jupyter nbconvert --to notebook --execute notebooks/titanic_eda.ipynb --output titanic_eda_executed.ipynb
```

## Expected outputs

- Inspection tables and data-quality observations.
- Before-and-after missing-value counts.
- `data/cleaned_train.csv`.
- Three required charts and the additional analysis charts in `charts/`.
- At least three data-supported observations and a concise conclusion.

## Conclusion

This project delivers the AVIP 2026 Task 2 Titanic EDA using only the Kaggle training dataset. It combines transparent cleaning, interpretable features, descriptive tables, and reproducible visualizations. The findings are observational and do not claim causation. No machine-learning model is used.
