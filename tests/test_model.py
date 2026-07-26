import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from trading_bot.model import ProbabilityModel


def test_probability_model_orders_columns():
    x = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 1, 0, 0]})
    y = np.array([0, 0, 1, 1])
    estimator = LogisticRegression().fit(x, y)
    raw = estimator.predict_proba(x)[:, 1]
    calibrator = LogisticRegression().fit(raw.reshape(-1, 1), y)
    model = ProbabilityModel(
        estimator=estimator,
        calibrator=calibrator,
        feature_names=["a", "b"],
        buy_threshold=0.6,
        sell_threshold=0.4,
        version="test",
        pair="XBTEUR",
        interval_minutes=15,
    )
    probability = model.predict_probability(x[["b", "a"]])
    assert probability.shape == (4,)
