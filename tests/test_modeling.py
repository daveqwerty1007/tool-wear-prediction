import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling import create_feature_target, train_random_forest, evaluate_accuracy


def test_create_feature_target_shape():
    df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6],
        'flank_wear': [0.1, 0.2, 0.3]
    })
    X, y = create_feature_target(df, features=['feat1', 'feat2'])
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_random_forest_accuracy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=['f1', 'f2'])
    df['flank_wear'] = y
    X_feat, y_target = create_feature_target(df, features=['f1', 'f2'])
    model, X_test, y_test = train_random_forest(X_feat, y_target, random_state=0)
    acc = evaluate_accuracy(model, X_test, y_test)
    assert acc > 0.8
