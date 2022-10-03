import pickle
import numpy as np
import pandas as pd
import warnings
import boa


warnings.filterwarnings("ignore")

items_y = pd.read_csv('data/newton_y_data.csv')
items_y = items_y.drop(columns=['Unnamed: 0'])

# Uncomment this to run use failed case only
items_y = items_y.iloc[8122:]

items_D = pd.read_csv('data/newton_D_data.csv')
items_D = items_D.drop(columns=['Unnamed: 0'])

contract = boa.load("../contracts/CurveCryptoMathOptimized3.vy")

gas_statistics_percents = []
gas_statistics_amount = []

remove_liquidity_tx_counter = []
failed_counter = []
newD0_fail_array = []

def calculate(idx):
    item = items_y.iloc[idx]
    tx_hash = item['tx']

    item_D_slice = items_D[items_D['tx']==tx_hash]
    if item_D_slice.shape[0] != 1:
        return

    ANN = int(item['ANN'])
    GAMMA = int(item['gamma'])
    x_original = [int(item['x0']), int(item['x1']), int(item['x2'])]
    D = int(item['D'])
    index = int(item['i'])
    x_copy = x_original[:]

    y, K0 = contract.get_y(ANN, GAMMA, x_original, D, index)
    y_gas = contract._computation.get_gas_used()
    x_copy[index] = y

    for i, item_D in item_D_slice.iterrows():
        x_unsorted = [int(item_D['x_unsorted_0']), int(item_D['x_unsorted_1']), int(item_D['x_unsorted_2'])]

        fee_alpha = x_unsorted[index]/int(item['output'])
        if x_unsorted[index]/int(item['output']) <= 1.:
            remove_liquidity_tx_counter.append(tx_hash)
            return

        D_orig = contract.newton_D(ANN, GAMMA, x_unsorted)
        D_orig_gas = contract._computation.get_gas_used()
        D_orig_new = contract.newton_D(ANN, GAMMA, x_unsorted, K0)
        D_orig_new_gas = contract._computation.get_gas_used()

        print(tx_hash)

        assert y/int(item['output']) == 1.
        assert x_copy[index] < x_unsorted[index]
        assert D_orig_new/int(item_D['output']) == 1.

        gas_statistics_amount.append((D_orig_gas, D_orig_new_gas))
    if (idx+1)%100 == 0:
        print(f'Gas statistics: {idx+1}\t{np.mean(gas_statistics_amount, axis=0).astype(int)}')

for idx in range(items_y.shape[0]):
    calculate(idx)