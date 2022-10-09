import boa
import pytest

INITIAL_PRICES = [21000, 1750]  # in usdt
INITIAL_DEPOSITS = 3 * 10**6  # 3M usdt worth


@pytest.fixture(scope="module", autouse=True)
def tricrypto_lp_token(deployer):
    with boa.env.prank(deployer):
        return boa.load(
            "contracts/CurveTokenV4.vy", "Curve USD-BTC-ETH", "crvUSDBTCETH"
        )


@pytest.fixture(scope="module")
def tricrypto_pool_init_params():
    yield {
        "A": 135 * 3**3 * 10000,
        "gamma": int(7e-5 * 1e18),
        "mid_fee": int(4e-4 * 1e10),
        "out_fee": int(4e-3 * 1e10),
        "allowed_extra_profit": 2 * 10**12,
        "fee_gamma": int(0.01 * 1e18),
        "adjustment_step": int(0.0015 * 1e18),
        "admin_fee": 0,
        "ma_half_time": 600,
    }


@pytest.fixture(scope="module")
def tricrypto_math(deployer):
    with boa.env.prank(deployer):
        return boa.load("contracts/CurveCryptoMathOptimized3.vy")


@pytest.fixture(scope="module", autouse=True)
def tricrypto_swap(
    owner,
    fee_receiver,
    tricrypto_pool_init_params,
    tricrypto_lp_token,
    tricrypto_math,
    coins,
    deployer,
):
    path = "contracts/CurveTricryptoOptimized.vy"
    with open(path, "r") as f:
        source = f.read()
        source = source.replace(
            "0x0000000000000000000000000000000000000000",
            tricrypto_math.address,
        )
        source = source.replace(
            "0x0000000000000000000000000000000000000001",
            tricrypto_lp_token.address,
        )

        source = source.replace(
            "0x0000000000000000000000000000000000000010", coins[0].address
        )
        source = source.replace(
            "0x0000000000000000000000000000000000000011", coins[1].address
        )
        source = source.replace(
            "0x0000000000000000000000000000000000000012", coins[2].address
        )

        source = source.replace(
            "1,#0", str(10 ** (18 - coins[0].decimals())) + ","
        )
        source = source.replace(
            "1,#1", str(10 ** (18 - coins[1].decimals())) + ","
        )
        source = source.replace(
            "1,#2", str(10 ** (18 - coins[2].decimals())) + ","
        )

    with boa.env.prank(deployer):
        yield boa.loads(
            source,
            owner,
            fee_receiver,
            tricrypto_pool_init_params["A"],
            tricrypto_pool_init_params["gamma"],
            tricrypto_pool_init_params["mid_fee"],
            tricrypto_pool_init_params["out_fee"],
            tricrypto_pool_init_params["allowed_extra_profit"],
            tricrypto_pool_init_params["fee_gamma"],
            tricrypto_pool_init_params["adjustment_step"],
            tricrypto_pool_init_params["admin_fee"],
            tricrypto_pool_init_params["ma_half_time"],
            INITIAL_PRICES,
        )


@pytest.fixture(scope="module")
def tricrypto2_swap_with_deposit(tricrypto_swap, token_minter, coins, user):
    quantities = []
    for idx, coin in enumerate(coins):
        quantity = (
            INITIAL_DEPOSITS[idx] / INITIAL_PRICES[idx] * 10 ** coin.decimals()
        )
        quantities.append(quantity)

        # mint coins for user:
        token_minter(coin, user, quantity)
        assert coin.balanceOf(user) == quantity

        # approve crypto_swap to trade coin for user:
        with boa.env.prank(user):
            coin.approve(tricrypto_swap, 2**256 - 1)

    # Very first deposit
    with boa.env.prank(user):
        tricrypto_swap.add_liquidity(quantities, 0)

    return tricrypto_swap
