from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_train.csv"
CHARTS_PATH = PROJECT_ROOT / "charts"

st.set_page_config(
    page_title="Titanic Dataset — Exploratory Data Analysis",
    page_icon="🚢",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the existing cleaned dataset without changing it."""
    return pd.read_csv(DATA_PATH)


def chart_path(filename: str) -> Path:
    """Return the path to an existing project chart."""
    return CHARTS_PATH / filename


df = load_data()

st.title("Titanic Dataset — Exploratory Data Analysis")
st.caption("AVIP 2026 | Data Science Task 2")
st.write(
    "An interactive view of the existing Titanic EDA project, focused on "
    "passenger survival patterns, data quality, and interpretable features. "
    "No machine-learning model is used."
)

with st.sidebar:
    st.header("Navigate")
    st.markdown(
        "[Dataset Overview](#dataset-overview)\n\n"
        "[Data Quality](#data-quality)\n\n"
        "[Feature Engineering](#feature-engineering)\n\n"
        "[Visualizations](#visualizations)\n\n"
        "[Key Findings](#key-findings)\n\n"
        "[Conclusion](#conclusion)"
    )
    st.divider()
    st.caption("Data source: existing data/cleaned_train.csv")

st.header("Dataset Overview", anchor="dataset-overview")
metric_columns = st.columns(3)
metric_columns[0].metric("Rows", f"{df.shape[0]:,}")
metric_columns[1].metric("Columns", df.shape[1])
metric_columns[2].metric("Survival rate", f"{df['Survived'].mean():.1%}")

st.subheader("Cleaned dataset")
st.dataframe(df, use_container_width=True, height=420)

st.subheader("Useful summary statistics")
st.dataframe(df.describe().T, use_container_width=True)

st.header("Data Quality", anchor="data-quality")
quality_columns = st.columns(3)
quality_columns[0].metric("Duplicate rows", int(df.duplicated().sum()))
quality_columns[1].metric("Missing cells", int(df.isna().sum().sum()))
quality_columns[2].metric("Cabin values known", f"{df['CabinKnown'].mean():.1%}")

st.subheader("Missing values")
missing_values = df.isna().sum().sort_values(ascending=False)
st.dataframe(
    missing_values.rename("missing_count").to_frame(),
    use_container_width=True,
)

st.subheader("Data types")
st.dataframe(
    df.dtypes.astype(str).rename("dtype").to_frame(),
    use_container_width=True,
)

st.info(
    "The notebook handled missing Age values with the median and missing "
    "Embarked values with the mode. Cabin identifiers were not fabricated; "
    "CabinKnown indicates whether cabin information was available."
)

st.header("Feature Engineering", anchor="feature-engineering")
feature_descriptions = {
    "FamilySize": "SibSp + Parch + 1; summarizes the passenger and accompanying family members.",
    "IsAlone": "Equals 1 when FamilySize is one, otherwise 0.",
    "AgeGroup": "Groups ages into child, teenager, young adult, adult, and senior categories.",
    "CabinKnown": "Equals 1 when a cabin value was recorded, otherwise 0.",
}

feature_cards = st.columns(4)
for column, card in zip(feature_descriptions, feature_cards):
    card.subheader(column)
    card.write(feature_descriptions[column])

st.subheader("Engineered feature sample")
st.dataframe(
    df[["FamilySize", "IsAlone", "AgeGroup", "CabinKnown"]].head(10),
    use_container_width=True,
)

st.header("Visualizations", anchor="visualizations")
st.write(
    "These are the existing chart exports from the EDA project. The Streamlit "
    "app reads them directly and does not recreate or modify them."
)

charts = [
    (
        "Survival by class and gender",
        "Compares observed survival rates across passenger classes and between female and male passengers.",
        "survival_by_class_gender.png",
    ),
    (
        "Passenger age distribution",
        "Shows how ages are distributed across the passenger population after the documented median-age handling.",
        "age_distribution.png",
    ),
    (
        "Numerical feature correlations",
        "Summarizes relationships among survival, class, age, family counts, fare, FamilySize, and IsAlone.",
        "correlation_heatmap.png",
    ),
    (
        "Additional distributions",
        "Provides context through overall survival, class-level survival, and fare distribution views.",
        "additional_distributions.png",
    ),
    (
        "Age groups and family size",
        "Explores survival-rate segmentation by age group and the distribution of family sizes across survival outcomes.",
        "age_group_and_family_size.png",
    ),
]

for index in range(0, len(charts), 2):
    chart_columns = st.columns(2)
    for column, (title, explanation, filename) in zip(
        chart_columns, charts[index : index + 2]
    ):
        with column:
            st.subheader(title)
            image_file = chart_path(filename)
            if image_file.exists():
                st.image(str(image_file), caption=filename, use_container_width=True)
            else:
                st.error(f"Chart file not found: {image_file}")
            st.caption(explanation)

st.header("Key Findings", anchor="key-findings")
st.write(
    "The following findings are carried over from the existing README and "
    "notebook. They describe associations in this dataset, not causation."
)
findings = [
    "Observed survival rates vary across passenger classes.",
    "Observed survival rates differ between female and male passengers, including within passenger classes.",
    "Age groups and family-size groups provide useful descriptive segmentation.",
    "Fare and passenger class are related and should be interpreted together.",
]
for finding in findings:
    st.markdown(f"- {finding}")

st.header("Conclusion", anchor="conclusion")
st.success(
    "This project delivers the AVIP 2026 Task 2 Titanic EDA using only the "
    "Kaggle training dataset. It combines transparent cleaning, interpretable "
    "features, descriptive tables, and reproducible visualizations. The "
    "findings are observational and do not claim causation. No machine-learning "
    "model is used."
)
