# FILE HAS BEEN MADE BY DIMGAR INC.


import functools
from math import factorial
import matplotlib.pyplot as plt

# Начальные параметры для Pi (n, lam, mu), pos определяется отдельно
args = [1, 1 / 7, 1 / 46]


# Функция для просчёта вероятности отказа и вероятность нахождения в состоянии S0
@functools.lru_cache()
def Pi(n, lam, mu):
    q = sum([lam ** i / (factorial(i) * mu ** i) for i in range(n + 1)])
    p0 = 1 / q
    p_otk = (lam ** n / (factorial(n) * mu ** n)) * p0
    return p_otk, p0


# Функция для просчёта математического ожидания
@functools.lru_cache()
def Mat(n, lam, mu, p0):
    m_n = 0
    for i in range(n + 1):
        m_n += i * p0 * (lam ** i / (factorial(i) * mu ** i))
    return m_n


# Массив графиков
ans_to_var = []
for i in range(3):
    ans_to_var.append([])

# Блок для просчета вероятности отказа и математического ожидания
pi = Pi(*args)
ans_to_var[0].append(pi[0])

mat = Mat(*args, pi[1])
ans_to_var[1].append(mat)
while pi[0] > 0.01:
    args[0] += 1
    pi = Pi(*args)
    ans_to_var[0].append(pi[0])

    mat = Mat(*args, pi[1])
    ans_to_var[1].append(mat)

# Блок для просчета коэффицента загруженности операторов
for i in range(1, len(ans_to_var[1]) + 1):
    ans_to_var[2].append(ans_to_var[1][i - 1] / i)

# Вывод графиков
for i in range(len(ans_to_var)):
    xpoints = [i for i in range(1, args[0] + 1)]
    ypoints = ans_to_var[i]
    plt.plot(xpoints, ypoints)
    plt.plot(xpoints, ypoints, 'o')
    plt.show()
    plt.close()
