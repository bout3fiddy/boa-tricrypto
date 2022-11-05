from datetime import timedelta

import boa
import pytest
import simulation_int_many as sim
from hypothesis import example, given, settings, note
from hypothesis import strategies as st
from vyper.utils import SizeLimits


N_COINS = 3
MAX_SAMPLES = 100000  # Increase for fuzzing

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
)
@example(
    A=2700,
    D=1000000000000020354,
    xD=90799920203792636,
    yD=10010000000000000,
    zD=501320194280195705,
    gamma=10000000000,
    j=1,
)
@settings(max_examples=MAX_SAMPLES, deadline=timedelta(seconds=1000))
def test_get_y(tricrypto_math, A, D, xD, yD, zD, gamma, j):
    X = [D * xD // 10**18, D * yD // 10**18, D * zD // 10**18]
    result_original = tricrypto_math.newton_y(A, gamma, X, D, j)
    try:
        (result_get_y, K0) = tricrypto_math.get_y_old(A, gamma, X, D, j)
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
    note("{\n"f"\t'ANN': {A},\n\t'D': {D},\n\t'xD': {xD},\n\t'yD': {yD},\n\t'zD': {zD},\n\t'GAMMA': {gamma},\n\t'index': {j}\n""}")
    # print(result_original)
    # print(result_get_y)
    assert abs(result_original - result_get_y) <= max(10**5, result_original/1e14)  # 10000 is $1e-14