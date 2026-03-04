import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    # Drop high-cardinality and unnecessary columns
    cols_to_remove = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Score", "Churn Reason"
    ]

    df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

    # Convert numeric columns safely
    numeric_cols = [
        "Total Charges",
        "CLTV",
        "Tenure Months",
        "Monthly Charges",
        "Senior Citizen"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Churn Value", "Churn Label"])
    y = df["Churn Value"]
    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame):

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def prepare_train_test_data(df: pd.DataFrame, test_size=0.2, random_state=42):

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_preprocessing_pipeline(X)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor