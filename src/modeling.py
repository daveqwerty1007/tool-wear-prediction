import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def create_feature_target(df: pd.DataFrame, features=None):
    """Return feature matrix X and target y for classification.

    If ``features`` is None, all columns except ``flank_wear`` are used as
    features. The target ``y`` is binarized based on the median of the
    ``flank_wear`` column if it is not already integer typed.
    """
    if features is None:
        features = [c for c in df.columns if c != "flank_wear"]
    X = df[features]
    y = df["flank_wear"]
    if not pd.api.types.is_integer_dtype(y) and not pd.api.types.is_bool_dtype(y):
        y = (y > y.median()).astype(int)
    return X, y


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric columns using :class:`StandardScaler`."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)


def train_random_forest(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                         random_state: int = 0):
    """Split ``X``/``y``, train a :class:`RandomForestClassifier` and return
    the fitted model along with the held‑out test arrays."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def evaluate_accuracy(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Return the classification accuracy for ``model`` on ``X_test``."""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
