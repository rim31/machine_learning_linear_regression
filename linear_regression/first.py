#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt


def firstprgm(km, t0, t1):
    try:
        km = float(km)
        t0 = float(t0)
        t1 = float(t1)
        result = t0 + (t1 * km)
        return result
    except ValueError:
        print("error value !")

def moindreCarre(a, b, km, price):
    totalError = 0
    for i in range(0, len(km)):
        totalError += (float(price[i]) - (a * float(km[i]) + b)) ** 2
    return totalError / float(len(km))

def calculGradient(a_current, b_current, km, price, learningRate):
    # b_gradient = 0
    # a_gradient = 0
    # N = float(len(km))
    # for i in range(0, len(km)):
    #     b_gradient += -(2/N) * (price[i] - ((a_current*km[i]) + b_current))
    #     a_gradient += -(2/N) * km[i] * (price[i] - ((a_current * km[i]) + b_current))
    # new_b = b_current - (learningRate * b_gradient)
    # new_a = a_current - (learningRate * a_gradient)
    # return [new_a, new_b]
    b_gradient = 0
    a_gradient = 0
    N = float(len(km))
    for i in range(0, len(km)):
        b_gradient += (1/N) * ( ((a_current*km[i]) + b_current) - price[i])
        a_gradient += (1/N) * km[i] * (((a_current * km[i]) + b_current) - price[i])
    new_b = b_current - (learningRate * b_gradient)
    new_a = a_current - (learningRate * a_gradient)
    return [new_a, new_b]



def calcultmp0(LR, a, b, km, price):
    tmp = 0.0
    j = 0
    for i in km:
        tmp =   1/(len(km)) * (firstprgm(km[j], a, b) - float(price[j])) + tmp
        j += 1
    # print(tmp)
    tmp0 = LR  *  tmp
    print(tmp0)
    return tmp0

# def calcultmp1(LR, a, b, km, price):
#     tmp = 0.0
#     j = 0
#     for i in km:
#         tmp = 1/(len(km)) * (firstprgm((km[j]-price[j]), a, b) * km[j]) + tmp
#         j += 1
#     print(tmp)
#     tmp1 = LR * tmp
#     # print(tmp1)
#     return tmp1

def first():
    # data = open("data.csv", "r")
    # contenu = data.read()
    # print(contenu)
    # data.close()
    i = 0
    km = []
    price = []
    with open("data.csv") as f :
       for line in f :
          if (i > 0):
            line = line.replace('\n', '')
            donnees =line.split(',')
            km.append(float(donnees[0]))
            price.append(float(donnees[1]))
          i = i + 1
# ========== Second partie =========
    learningRate = 0.01
    # calcul de theta 0 et 1
    # ======================
    a = (price[0] - price[len(price) -1])/(km[0] - km[len(km)-1])
    b = price[0] - a * km[0]
    print(a, b)
    print("Methode moindre carre \n=======================")
    print(moindreCarre(a, b, km, price))
    learningRate = 0.01
    result = []
    result = calculGradient(a, b, km, price, learningRate)
    print(result)
    print("Methode moindre carre \n=======================")
    print(moindreCarre(result[0], result[1], km, price))

    tmp0 = calcultmp0(learningRate, a, b, km, price)
    # tmp1 = calcultmp1(learningRate, a, b, km, price)
    # t0 = learningRate * 1/len(km) * np.sum()
# ========= BONUS =========
    plt.plot(km, price, 'ro')
    plt.ylabel('Price (euro))')
    plt.xlabel('Kilometrage (km)')
    plt.plot([0.0, 250000.0], [8500.0, 3500.0], 'b--', lw=2) # Red straight line
    # plt.savefig('StraightLine.png')
    # plt.show()

# def test():
#     X = 2 * np.random.rand(100, 1)
#     y = 4 + 3 * X + np.random.rand(100, 1)
#
#     X_b = np.c_[np.ones((100, 1)), X]
#     theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#
#     X_new = np.array([[0], [2]])
#     X_new_b = np.c_[np.ones((2, 1)), X_new]
#     y_predict = X_new_b.dot(theta_best)
#
#     plt.figure(figsize=(10, 10))
#     predictions = plt.plot(X_new, y_predict, "r-")
#     plt.plot(X, y, "b.")
#     plt.axis([0, 2, 0, 15])
#     plt.figlegend(predictions, "pred", 'upper center')
#     plt.show()

def premierPgrm(km):
    t0 = np.loadtxt("theta0.txt", unpack='true')
    t1 = np.loadtxt("theta1.txt", unpack='true')
    # print(t0, t1)
    print("penser a verifier les coef t0.txt et t1.txt")
    print("your car worth ")
    price = t0 + (t1 * km)
    print(price)
    print(firstprgm(km, t0, t1))
    return (price)



def main(argv):
    if __name__ == "__main__":
        main(sys.argv[1:])
# print("please enter a kilometrage")
if len(sys.argv) == 2:
    try:
        km = float(sys.argv[1])
        premierPgrm(km)
    except ValueError:
        print("That's not an int!")
elif len(sys.argv) == 1:
    first()
    # test()
