import pytest
import numpy as np
from src.geometric_brownian_motion import GeometricBrownianMotion

def test_gbm_shape():
    model = GeometricBrownianMotion(S0=100, drift=0.05, volatility=0.2, maturity=1, n_steps=252)
    paths = model.generate_paths(1000)
    # shape => (1000, 253)
    assert paths.shape == (1000, 253)