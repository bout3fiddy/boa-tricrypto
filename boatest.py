import boa
from itertools import permutations


ctr = boa.load("contracts/CurveCryptoMathOptimized3.vy")
print(len(ctr.bytecode))

# i, j, k = list(permutations(range(3)))[perm]
# X = [x, x * yx // 10**18, x * zx // 10**18]
# X = [X[i], X[j], X[k]]
# res_sim = sim.solve_D(A, gamma, X)
# res_newton = ctr.newton_D_original(A, gamma, X)

# # X = [D * xD // 10**18, D * yD // 10**18, D * zD // 10**18]
# # res_sim = sim.solve_x(A, gamma, X, D, j)
# # res_newton = ctr.newton_y(A, gamma, X, D, j)
# # (res_get_y, K0) = ctr.get_y(A, gamma, X, D, j)

# print(
#         f'\n'
#         # f'Sim A               :  \t{sim.solve_x(A, gamma, X, D, j)}\n'
#         # f'Sim A * 3**3 * 10000:  \t{sim.solve_x(A * 3**3 * 10000, gamma, X, D, j)}\n'
#         # f'Newton_y:        \t{res_newton}\n'
#         # f'get_y:           \t{res_get_y}\n'
#         # f'Diff Sim-Newton: \t{abs(res_sim - res_newton)}\n'
#         f'Sim Newton_D:        \t{res_sim}\n'
#         f'Newton_D:            \t{res_newton}\n'
#     )
