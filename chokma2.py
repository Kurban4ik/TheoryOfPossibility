# FILE HAS BEEN MADE BY DIMGAR INC.

import functools
from math import factorial
import matplotlib.pyplot as plt

# Начальные параметры для Pi ( n, m, lam, mu), pos определяется отдельно
args = [45, 1, 1 / 66, 1 / 42]


# Функция для просчёта вероятности отказа
@functools.lru_cache()
def Pi(n, m, lam, mu, pos=1):
    q = 1 + sum([(lam ** i) / ((mu ** i) * factorial(i)) * factorial(n) / factorial(n - i) for i in range(1, m + 1)])
    q += sum([(lam ** i) / ((mu ** i) * factorial(m) * m ** (i - m)) * factorial(n) / factorial(n - i) for i in
              range(m + 1, n + 1)])
    p0 = 1 / q

    if 1 <= pos <= m:
        return p0 * (lam ** pos) / ((mu ** pos) * factorial(pos)) * factorial(n) / factorial(n - pos)
    elif 1 <= pos:
        return p0 * (lam ** pos) / ((mu ** pos) * factorial(m) * m ** (pos - m)) * factorial(n) / factorial(n - pos)
    return p0


# Массив графиков
ans_to_var = []
for i in range(5):
    ans_to_var.append([])

# Блок для подсчета математического ожидания числа простаивающих станков (0)
for i in range(1, args[0] + 1):
    mat_it = 0
    args[1] = i
    for j in range(args[0] + 1):
        mat_it += Pi(*args, j) * j
    ans_to_var[0].append(mat_it)

# Блок для подсчета математического ожидания числа станков, ожидающих обслуживания (1)
# Блок для подсчета вероятности ожидания обслуживания (2)
for i in range(1, args[0] + 1):
    matq = 0
    pq = 0
    args[1] = i
    for j in range(1, args[0] - i + 1):
        pq += Pi(*args, i + j)
        matq += Pi(*args, i + j) * j
    ans_to_var[1].append(matq)
    ans_to_var[2].append(pq)

# Блок для подсчета математического ожидания числа занятых наладчиков (3)
# Блок для подсчета коэффициента занятости наладчиков (4)
for i in range(1, args[0] + 1):
    mat_m = 0
    args[1] = i
    for j in range(i + 1):
        mat_m += Pi(*args, j) * j
    args[1] = i
    for j in range(1, args[0] - i + 1):
        mat_m += Pi(*args, i + j) * i
    ans_to_var[3].append(mat_m)
    ans_to_var[4].append(mat_m / i)

# Вывод графиков
for i in range(len(ans_to_var)):
    xpoints = [i for i in range(1, args[0] + 1)]
    ypoints = ans_to_var[i]
    plt.plot(xpoints, ypoints)
    plt.plot(xpoints, ypoints, 'o')
    plt.show()
    plt.close()
