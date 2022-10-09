# @version 0.3.7
# (c) Curve.Fi, 2022
# Math for USDT/BTC/ETH pool


N_COINS: constant(uint256) = 3  # <- change
A_MULTIPLIER: constant(uint256) = 10000

MIN_GAMMA: constant(uint256) = 10**10
MAX_GAMMA: constant(uint256) = 5 * 10**16

MIN_A: constant(uint256) = N_COINS**N_COINS * A_MULTIPLIER / 100
MAX_A: constant(uint256) = N_COINS**N_COINS * A_MULTIPLIER * 1000


# --- Internal maff ---

@internal
@pure
def cbrt(x: uint256) -> uint256:

    # we artificially set a cap to the values for which we can compute the
    # cube roots safely. This is not to say that there are no values above
    # max(uint256) // 10**36 for which we cannot get good cube root estimates.
    # However, beyond this point, accuracy is not guaranteed since overflows
    # start to occur.
    assert x < 115792089237316195423570985008687907853269, "inaccurate cbrt"  # TODO: check limits again

    # we increase precision of input `x` by multiplying 10 ** 36.
    # in such cases: cbrt(10**18) = 10**18, cbrt(1) = 10**12
    xx: uint256 = unsafe_mul(x, 10**36)

    # get log2(x) for approximating initial value
    # logic is: cbrt(a) = cbrt(2**(log2(a))) = 2**(log2(a) / 3) ≈ 2**|log2(a)/3|
    # from: https://github.com/transmissions11/solmate/blob/b9d69da49bbbfd090f1a73a4dba28aa2d5ee199f/src/utils/FixedPointMathLib.sol#L352

    a_pow: int256 = 0
    if xx > 340282366920938463463374607431768211455:
        a_pow = 128
    if unsafe_div(xx, shift(2, a_pow)) > 18446744073709551615:
        a_pow = a_pow | 64
    if unsafe_div(xx, shift(2, a_pow)) > 4294967295:
        a_pow = a_pow | 32
    if unsafe_div(xx, shift(2, a_pow)) > 65535:
        a_pow = a_pow | 16
    if unsafe_div(xx, shift(2, a_pow)) > 255:
        a_pow = a_pow | 8
    if unsafe_div(xx, shift(2, a_pow)) > 15:
        a_pow = a_pow | 4
    if unsafe_div(xx, shift(2, a_pow)) > 3:
        a_pow = a_pow | 2
    if unsafe_div(xx, shift(2, a_pow)) > 1:
        a_pow = a_pow | 1

    # initial value: 2**|log2(a)/3|
    # which is: 2 ** (n / 3) * 1260 ** (n % 3) / 1000 ** (n % 3)
    a_pow_mod: uint256 = convert(a_pow, uint256) % 3
    a: uint256 = unsafe_div(
        unsafe_mul(
            pow_mod256(
                2,
                unsafe_div(
                    convert(a_pow, uint256), 3
                )
            ),
            pow_mod256(1260, a_pow_mod)
        ),
        pow_mod256(1000, a_pow_mod)
    )

    # 7 newton raphson iterations:
    a = unsafe_div(unsafe_add(unsafe_mul(2, a), unsafe_div(xx, unsafe_mul(a, a))), 3)
    a = unsafe_div(unsafe_add(unsafe_mul(2, a), unsafe_div(xx, unsafe_mul(a, a))), 3)
    a = unsafe_div(unsafe_add(unsafe_mul(2, a), unsafe_div(xx, unsafe_mul(a, a))), 3)
    a = unsafe_div(unsafe_add(unsafe_mul(2, a), unsafe_div(xx, unsafe_mul(a, a))), 3)
    a = unsafe_div(unsafe_add(unsafe_mul(2, a), unsafe_div(xx, unsafe_mul(a, a))), 3)
    a = unsafe_div(unsafe_add(unsafe_mul(2, a), unsafe_div(xx, unsafe_mul(a, a))), 3)
    a = unsafe_div(unsafe_add(unsafe_mul(2, a), unsafe_div(xx, unsafe_mul(a, a))), 3)

    return a


@internal
@pure
def exp(_power: int256) -> uint256:

    if _power <= -42139678854452767551:
        return 0

    if _power >= 135305999368893231589:
        raise "exp overflow"

    x: int256 = unsafe_div(unsafe_mul(_power, 2**96), 10**18)

    k: int256 = unsafe_div(
        unsafe_add(
            unsafe_div(unsafe_mul(x, 2**96), 54916777467707473351141471128),
            2**95
        ),
        2**96
    )
    x = unsafe_sub(x, unsafe_mul(k, 54916777467707473351141471128))

    y: int256 = unsafe_add(x, 1346386616545796478920950773328)
    y = unsafe_add(unsafe_div(unsafe_mul(y, x), 2**96), 57155421227552351082224309758442)
    p: int256 = unsafe_sub(unsafe_add(y, x), 94201549194550492254356042504812)
    p = unsafe_add(unsafe_div(unsafe_mul(p, y), 2**96), 28719021644029726153956944680412240)
    p = unsafe_add(unsafe_mul(p, x), (4385272521454847904659076985693276 * 2**96))

    q: int256 = x - 2855989394907223263936484059900
    q = unsafe_add(unsafe_div(unsafe_mul(q, x), 2**96), 50020603652535783019961831881945)
    q = unsafe_sub(unsafe_div(unsafe_mul(q, x), 2**96), 533845033583426703283633433725380)
    q = unsafe_add(unsafe_div(unsafe_mul(q, x), 2**96), 3604857256930695427073651918091429)
    q = unsafe_sub(unsafe_div(unsafe_mul(q, x), 2**96), 14423608567350463180887372962807573)
    q = unsafe_add(unsafe_div(unsafe_mul(q, x), 2**96), 26449188498355588339934803723976023)

    return shift(
        unsafe_mul(convert(unsafe_div(p, q), uint256), 3822833074963236453042738258902158003155416615667),
        unsafe_sub(k, 195))


# --- External maff functions ---


# TODO: the following method should use cbrt:
@external
@view
def geometric_mean(unsorted_x: uint256[3], sort: bool = True) -> uint256:
    """
    @notice calculates geometric of 3 element arrays: cbrt(x[0] * x[1] * x[2])
    @dev This approach is specifically optimised for 3 element arrays. To
         use it for 2 element arrays, consider using the vyper builtin: isqrt.
    @param unsorted_x: array of 3 uint256 values
    @param sort: if True, the array will be sorted before calculating the mean
    @return the geometric mean of the array
    """
    x: uint256[3] = unsorted_x

    # cheap sort using temp var: only works if N_COINS == 3
    if sort:
        temp_var: uint256 = x[0]
        if x[0] < x[1]:
            x[0] = x[1]
            x[1] = temp_var
        if x[0] < x[2]:
            temp_var = x[0]
            x[0] = x[2]
            x[2] = temp_var
        if x[1] < x[2]:
            temp_var = x[1]
            x[1] = x[2]
            x[2] = temp_var

    # geometric mean calculation:
    D: uint256 = x[0]
    diff: uint256 = 0
    for i in range(255):
        D_prev: uint256 = D
        tmp: uint256 = 10**18
        for _x in x:
            tmp = tmp * _x / D
        D = D * ((N_COINS - 1) * 10**18 + tmp) / (N_COINS * 10**18)
        if D > D_prev:
            diff = D - D_prev
        else:
            diff = D_prev - D
        if diff <= 1 or diff * 10**18 < D:
            return D
    raise "Did not converge"


@external
@view
def halfpow(power: uint256) -> uint256:
    """
    1e18 * 0.5 ** (power/1e18)

    Inspired by: https://github.com/transmissions11/solmate/blob/4933263adeb62ee8878028e542453c4d1a071be9/src/utils/FixedPointMathLib.sol#L34

    This should cost about 1k gas
    """

    # TODO: borrowed from unoptimised halfpow, please check the following:
    if unsafe_div(power, 10**18) > 59:
        return 0

    # exp(-ln(2) * x) = 0.5 ** x. so, get -ln(2) * x:
    return self.exp(-1 * 693147180559945344 * convert(power, int256) / 10 ** 18)


@external
@view
def get_D(ANN: uint256, gamma: uint256, x_unsorted: uint256[N_COINS]) -> uint256:
    """
    Finding the invariant analytically.
    ANN is higher by the factor A_MULTIPLIER
    ANN is already A * N**N
    """
    #TODO: add tricrypto math optimisations here
    return ANN


@external
@view
def get_y(ANN: uint256, gamma: uint256, x: uint256[N_COINS], D: uint256, i: uint256) -> uint256[2]:
    """
    Calculating x[i] given other balances x[0..N_COINS-1] and invariant D
    ANN = A * N**N
    """

    j: uint256 = 0
    k: uint256 = 0
    if i == 0:
        j = 1
        k = 2
    elif i == 1:
        j = 0
        k = 2
    elif i == 2:
        j = 0
        k = 1

    a: uint256 = 10**28/27
    b: uint256 = unsafe_sub(
        unsafe_add(
            unsafe_div(
                unsafe_mul(2*10**10, gamma), 27
            ), 10**28/9    
        ),  
        unsafe_div(
            unsafe_div(
                unsafe_div(
                    unsafe_mul(
                            unsafe_div(
                                    unsafe_mul(
                                        unsafe_div(D**2, x[j]), gamma**2
                                    ), x[k]
                                ), ANN
                        ), 27**2
                ), 10**8
            ), A_MULTIPLIER
        )
    )
    c: uint256 = 0
    if D > x[j] + x[k]:
        c = unsafe_sub(
                unsafe_add(
                    unsafe_div(
                        unsafe_div(
                            unsafe_mul(gamma, 
                                unsafe_add(gamma, 4*10**18)
                                ), 27
                        ), 10**8
                    ), 10**28/9
                ),
                unsafe_div(
                    unsafe_div(
                        unsafe_div(
                            unsafe_mul(
                                    unsafe_div(
                                        unsafe_mul(
                                            gamma**2, unsafe_sub(
                                                unsafe_sub(D, x[j]), x[k]
                                            )
                                        ), D
                                ), ANN
                            ), 10**8
                        ), 27
                    ), A_MULTIPLIER
                )
            )
    else:
        c = unsafe_add(
                unsafe_add(
                    unsafe_div(
                        unsafe_div(
                            unsafe_mul(gamma, 
                                unsafe_add(gamma, 4*10**18)
                                ), 27
                        ), 10**8
                    ), 10**28/9
                ),
                unsafe_div(
                    unsafe_div(
                        unsafe_div(
                            unsafe_mul(
                                    unsafe_div(
                                        unsafe_mul(
                                            gamma**2, unsafe_sub(
                                                unsafe_add(x[j], x[k]), D
                                            )
                                        ), D
                                ), ANN
                            ), 10**8
                        ), 27
                    ), A_MULTIPLIER
                )
            )
    d: uint256 = unsafe_div(
        unsafe_div(
            unsafe_add(10**18, gamma)**2, 10**8
        ), 27
    )

    delta0: uint256 = unsafe_sub(
        unsafe_div(
            unsafe_mul(
                unsafe_mul(3, a), c
            ), b
        ), b
    )
    delta1: uint256 = unsafe_sub(
        unsafe_sub(
            unsafe_div(
                    unsafe_mul(
                        unsafe_mul(9, a), c
                    ), b
                ), unsafe_mul(2, b)
        ), unsafe_div(
            unsafe_mul(
                unsafe_div(
                    unsafe_mul(27, a**2), b
                ), d
            ), b
        )
    )

    C1: uint256 = self.cbrt(
        unsafe_div(
            unsafe_mul(
                    unsafe_div(
                            unsafe_div(
                                unsafe_mul(
                                    b, unsafe_add(
                                        delta1, isqrt(
                                            unsafe_add(
                                                delta1**2, unsafe_div(
                                                    unsafe_mul(
                                                        4, delta0**3
                                                    ), b
                                                )
                                            )
                                        )
                                    )
                                ), 2
                            ), 10**18
                        ), b
                ), 10**18
        )
    ) * 10**6
    root_K0: uint256 = unsafe_div(
        unsafe_add(
            unsafe_sub(
                unsafe_mul(10**18, b),
                unsafe_mul(10**18, C1)
            ), unsafe_div(
                unsafe_mul(
                    unsafe_mul(10**18, b), delta0
                ), C1
            )
        ), 3*a
    )

    return [
        unsafe_div(
            unsafe_div(
                unsafe_mul(
                    unsafe_div(
                        unsafe_mul(
                            unsafe_div(
                                unsafe_mul(root_K0, D), x[j]
                            ), D
                        ), x[k]
                    ), D
                ), 27
            ), 10**18
        ),
        root_K0
        ]


### Functions below should be used in this branch only

@internal
@pure
def sort(A0: uint256[N_COINS]) -> uint256[N_COINS]:
    """
    Insertion sort from high to low
    """
    A: uint256[N_COINS] = A0
    for i in range(1, N_COINS):
        x: uint256 = A[i]
        cur: uint256 = i
        for j in range(N_COINS):
            y: uint256 = A[cur-1]
            if y > x:
                break
            A[cur] = y
            cur -= 1
            if cur == 0:
                break
        A[cur] = x
    return A

@internal
@view
def _geometric_mean(unsorted_x: uint256[N_COINS], sort: bool = True) -> uint256:
    """
    (x[0] * x[1] * ...) ** (1/N)
    """
    x: uint256[N_COINS] = unsorted_x
    if sort:
        x = self.sort(x)
    D: uint256 = x[0]
    diff: uint256 = 0
    for i in range(255):
        D_prev: uint256 = D
        tmp: uint256 = 10**18
        for _x in x:
            tmp = tmp * _x / D
        D = D * ((N_COINS - 1) * 10**18 + tmp) / (N_COINS * 10**18)
        if D > D_prev:
            diff = D - D_prev
        else:
            diff = D_prev - D
        if diff <= 1 or diff * 10**18 < D:
            return D
    raise "Did not converge"

@external
@view
def newton_D(ANN: uint256, gamma: uint256, x_unsorted: uint256[N_COINS], K0_prev: uint256 = 0) -> uint256:
    """
    Finding the invariant using Newton method.
    ANN is higher by the factor A_MULTIPLIER
    ANN is already A * N**N

    Currently uses 60k gas
    """
    # Safety checks
    assert ANN > MIN_A - 1 and ANN < MAX_A + 1  # dev: unsafe values A
    assert gamma > MIN_GAMMA - 1 and gamma < MAX_GAMMA + 1  # dev: unsafe values gamma

    # Initial value of invariant D is that for constant-product invariant
    # x: uint256[N_COINS] = self.sort(x_unsorted)
    x: uint256[N_COINS] = x_unsorted

    assert x[0] > 10**9 - 1 and x[0] < 10**15 * 10**18 + 1  # dev: unsafe values x[0]
    for i in range(1, N_COINS):
        frac: uint256 = x[i] * 10**18 / x[0]
        assert frac > 10**11-1  # dev: unsafe values x[i]

    S: uint256 = 0
    for x_i in x:
        S += x_i 

    D: uint256 = 0
    if K0_prev == 0:
        D = N_COINS * self._geometric_mean(x, False)
    else:
        D = self.cbrt(x_unsorted[0]*x_unsorted[1]/10**18*x_unsorted[2]*27/K0_prev)*10**6

    for i in range(255):
        D_prev: uint256 = D

        K0: uint256 = 10**18
        for _x in x:
            K0 = K0 * _x * N_COINS / D

        _g1k0: uint256 = gamma + 10**18
        if _g1k0 > K0:
            _g1k0 = _g1k0 - K0 + 1
        else:
            _g1k0 = K0 - _g1k0 + 1

        # D / (A * N**N) * _g1k0**2 / gamma**2
        mul1: uint256 = 10**18 * D / gamma * _g1k0 / gamma * _g1k0 * A_MULTIPLIER / ANN

        # 2*N*K0 / _g1k0
        mul2: uint256 = (2 * 10**18) * N_COINS * K0 / _g1k0

        neg_fprime: uint256 = (S + S * mul2 / 10**18) + mul1 * N_COINS / K0 - mul2 * D / 10**18

        # D -= f / fprime
        D_plus: uint256 = D * (neg_fprime + S) / neg_fprime
        D_minus: uint256 = D*D / neg_fprime
        if 10**18 > K0:
            D_minus += D * (mul1 / neg_fprime) / 10**18 * (10**18 - K0) / K0
        else:
            D_minus -= D * (mul1 / neg_fprime) / 10**18 * (K0 - 10**18) / K0

        if D_plus > D_minus:
            D = D_plus - D_minus
        else:
            D = (D_minus - D_plus) / 2

        diff: uint256 = 0
        if D > D_prev:
            diff = D - D_prev
        else:
            diff = D_prev - D

        if diff * 10**14 < max(10**16, D):  # Could reduce precision for gas efficiency here
            # Test that we are safe with the next newton_y
            for _x in x:
                frac: uint256 = _x * 10**18 / D
                assert (frac > 10**16 - 1) and (frac < 10**20 + 1)  # dev: unsafe values x[i]
            return D

    raise "Did not converge"