import boa
import pytest
import simulation_int_many as sim
from hypothesis import example, given, settings, note
from hypothesis import strategies as st
from vyper.utils import SizeLimits
from datetime import timedelta
from decimal import Decimal


N_COINS = 3
MAX_SAMPLES = 1000000  # Increase for fuzzing

A_MUL = 10000 * 3**3
MIN_A = int(0.01 * A_MUL)
MAX_A = 1000 * A_MUL

# gamma from 1e-8 up to 0.05
MIN_GAMMA = 10**10
MAX_GAMMA = 5 * 10**16

@given(
       A=st.integers(min_value=MIN_A, max_value=MAX_A),
       D=st.integers(min_value=10**18, max_value=10**14 * 10**18),  # 1 USD to 100T USD
       xD=st.integers(min_value=int(1.001e16), max_value=int(0.999e20)),  # <- ratio 1e18 * x/D, typically 1e18 * 1
       yD=st.integers(min_value=int(1.001e16), max_value=int(0.999e20)),  # <- ratio 1e18 * y/D, typically 1e18 * 1
       zD=st.integers(min_value=int(1.001e16), max_value=int(0.999e20)),  # <- ratio 1e18 * z/D, typically 1e18 * 1
       gamma=st.integers(min_value=MIN_GAMMA, max_value=MAX_GAMMA),
       j=st.integers(min_value=0, max_value=2),
       ethScalePrice=st.integers(min_value=10**2, max_value=10**4),
       btcScalePrice=st.integers(min_value=10**4, max_value=10**5),
       mid_fee=st.sampled_from((0.7e-3, 1e-3, 1.2e-3, 4e-3)),
       out_fee=st.sampled_from((4.0e-3, 10.0e-3)),
       fee_gamma=st.sampled_from((int(1e-2 * 1e18), int(2e-6 * 1e18))),
)

@settings(max_examples=MAX_SAMPLES, deadline=timedelta(seconds=1000))
def test_get_y(tricrypto_math, A, D, xD, yD, zD, gamma, j, ethScalePrice, btcScalePrice, mid_fee, out_fee, fee_gamma):
    X = [D * xD // 10**18, D * yD // 10**18, D * zD // 10**18]

    try:
        (result_get_y, K0) = tricrypto_math.get_y_safe_int(A, gamma, X, D, j)
    except Exception:
        # May revert is the state is unsafe for the next time
        safe = all(f >= 1.1e16 and f <= 0.9e20 for f in [_x * 10**18 // D for _x in X])
        XX = X[:]
        XX[j] = result_original
        safe &= all(f >= 1.1e16 and f <= 0.9e20 for f in [_x * 10**18 // D for _x in XX])
        if safe:
            raise
        else:
            return
    note("{"f"'ANN': {A}, 'D': {D}, 'xD': {xD}, 'yD': {yD}, 'zD': {zD}, 'GAMMA': {gamma}, 'index': {j}""}\n")

    price_scale = (1, ethScalePrice, btcScalePrice)
    y = X[j]
    dy = X[j] - result_get_y
    dy -= 1

    if j > 0:
        dy = dy * 10**18 // price_scale[j-1]

    fee = sim.get_fee(X, fee_gamma, mid_fee, out_fee)
    dy -= fee * dy // 10**10
    y -= dy
    X[j] = y

    print(f'Fee: {fee}\n')
    assert fee != 0.

    result_sim = tricrypto_math.newton_D(A, gamma, X)
    if all(f >= 1.1e16 and f <= 0.9e20 for f in [_x * 10**18 / result_sim for _x in X]):
        result_contract = tricrypto_math.newton_D(A, gamma, X, K0)
        assert abs(result_sim - result_contract) <= max(1000, result_sim/1e15)