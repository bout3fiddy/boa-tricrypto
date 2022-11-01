import boa
import pytest
import simulation_int_many as sim
from hypothesis import example, given, settings
from hypothesis import strategies as st
from vyper.utils import SizeLimits
from datetime import timedelta
from itertools import permutations


N_COINS = 3
MAX_SAMPLES = 50  # Increase for fuzzing

A_MUL = 10000 * 3**3
MIN_A = int(0.01 * A_MUL)
MAX_A = 1000 * A_MUL

# gamma from 1e-8 up to 0.05
MIN_GAMMA = 10**10
MAX_GAMMA = 5 * 10**16


@given(
       A=st.integers(min_value=MIN_A, max_value=MAX_A),
       x=st.integers(min_value=10**9, max_value=10**14 * 10**18),  # 1e-9 USD to 100T USD
       yx=st.integers(min_value=int(1.1e11), max_value=10**18),  # <- ratio 1e18 * y/x, typically 1e18 * 1
       zx=st.integers(min_value=int(1.1e11), max_value=10**18),  # <- ratio 1e18 * z/x, typically 1e18 * 1
       perm=st.integers(min_value=0, max_value=5),  # Permutation
       gamma=st.integers(min_value=MIN_GAMMA, max_value=MAX_GAMMA)
)
@example(
    A=2700,
    x=1000000000,
    yx=194268000000000,
    zx=261522000000000,
    perm=2,
    gamma=10000000000,
)
@settings(max_examples=MAX_SAMPLES)
def test_newton_D(tricrypto_math, A, x, yx, zx, perm, gamma):
    i, j, k = list(permutations(range(3)))[perm]
    X = [x, x * yx // 10**18, x * zx // 10**18]
    X = [X[i], X[j], X[k]]
    result_sim = sim.solve_D(A, gamma, X)
    if all(f >= 1.1e16 and f <= 0.9e20 for f in [_x * 10**18 / result_sim for _x in X]):
        result_contract = tricrypto_math.newton_D(A, gamma, X)
        result_contract_newD0 = tricrypto_math.newton_D(A, gamma, X)
        assert abs(result_sim - result_contract) <= max(1000, result_sim/1e15)  # 1000 is $1e-15