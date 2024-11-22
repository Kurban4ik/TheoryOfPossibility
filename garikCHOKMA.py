import functools
from math import factorial
import matplotlib.pyplot as plt
import numpy as np

args = [1 / 7, 1 / 46, 20, 0]


# Функция для просчёта вероятности отказа
@functools.lru_cache()
def Pi(lam, mu, n, m, pos=None):
    """
    :param lam: -
    :param mu: -
    :param n: amount of factory lines
    :param m: query amount
    :param pos: if None returns possibility of reject else possibility of pos position
    :return: possibility of reject
    """
    p0 = 0
    ro = lam / mu

    for i in range(n + 1):
        p0 += ro ** i / factorial(i)
    p0 = 1 / (p0 + (ro ** n / factorial(n)) * sum([(ro ** j / n ** j) for j in range(1, m + 1)]))
    if pos == None:
        return (ro ** (n + m) / (factorial(n) * n ** m)) * p0
    return (ro ** pos / (factorial(min(n, pos)) * n ** max(0, pos - n))) * p0

# Начальные параметры для Pi (lam, mu, n, m), pos определяется отдельно
args1 = args.copy()
args2 = args.copy()


# Ответы в формате вида (значения графика по икс: list, значения графика по игрек: list)
ans_to_var_n = []
ans_to_var_m = []

# Блок просчёта вероятности отказа
ans_to_var_n.append([])
ans_to_var_m.append([])
for i in range(1, 20 + 1):
    p_otk_ni = []
    p_otk_mi = []
    args1[2] = i
    args2[3] = i
    for j in range(1, 20 + 1):
        args1[3] = j
        args2[2] = j
        p_otk_ni.append(Pi(*args1))
        p_otk_mi.append(Pi(*args2))
    ans_to_var_n[0].append((range(1, 20 + 1), p_otk_ni))
    ans_to_var_m[0].append((range(1, 20 + 1), p_otk_mi))

# Блок просчёта матожидания
ans_to_var_n.append([])
ans_to_var_m.append([])

for var_n in range(1, 20 + 1):
    args1[2] = var_n
    ys = []
    for var_m in range(21):
        args1[3] = var_m
        ys.append(sum([Pi(*args1, k) * min(k, var_n) for k in range(var_n + var_m + 1 )]))
    plt.plot(list(range(21)), ys)

plt.show()
plt.close()
