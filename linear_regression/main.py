#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt


def firstprgm(km, t0, t1):
    if (km <= 0):
        print("Dosen't exist !")
        return
    try:
        km = float(km)
        t0 = float(t0)
        t1 = float(t1)
        result = float(t0 + (t1 * km))
        if result < 0:
            result = 0
        return result
    except ValueError:
        print("error value !")



def moindreCarre(a, b, km, price):
    totalError = 0
    for i in range(0, len(km)):
        totalError += (float(price[i]) - (a * float(km[i]) + b)) ** 2
    return totalError / float(len(km))



def calculGradient(a_current, b_current, km, price, learningRate):
    bonus_secondprgoram(a_current, b_current)
    iterations = 1000
    print(a_current, b_current)
    for j in range(0, iterations):
        b_gradient = 0
        a_gradient = 0
        m = float(len(km))
        for i in range(0, len(km)):
            b_gradient += (1/m) * ( ((a_current*km[i]) + b_current) - price[i])
            a_gradient += (1/m) * km[i] * (((a_current * km[i]) + b_current) - price[i])
        new_a = a_current
        new_b = b_current
        new_b = b_current - (learningRate * b_gradient)
        new_a = a_current - (learningRate * a_gradient)
        if (j % 500) == 0:
            bonus_secondprgoram(new_a, new_b)
            print(new_a, new_b)

    # m = len(km)
    # for i in range(iterations):
    #     sumDiff0 = 0
    #     sumDiff1 = 0
    #     for j in range(len(km)):
    #         # //normalisation
    #         # sumDiff0 += a_current + b_current * km[j] - price[j]
    #         # sumDiff1 += (a_current + b_current * km[j] - price[j]) * km[j]
    #         sumDiff0 +=  ( ((a_current*km1) + b_current) - price1)
    #         sumDiff1 +=  km1 * (((a_current * km1) + b_current) - price1)
    #     a_current = a_current - learningRate * sumDiff0 / m
    #     b_current = b_current - learningRate * sumDiff1 / m
    #     if (j % 500) == 0:
    #         bonus_secondprgoram(a_current, b_current)
    #         print(a_current, b_current)

    return(new_a, new_b)


def parse():
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
    return(km, price)

def yaxb(km, price):
    # ========  calcul de theta 0 et 1  ===========
    a = (price[0] - price[len(price) -1])/(km[0] - km[len(km)-1])
    b = price[0] - a * km[0]
    # bonus_estimation(a, b)
    secontPrgm(a, b, km, price, 0.1)
    return(a, b)

def secontPrgm(a, b, km, price, learningRate):
    # print("Methode moindre carre \n=======================")
    # print(moindreCarre(a, b, km, price))
    res = []
    res = calculGradient(a, b, km, price, learningRate)
    print(res)
    # print("Methode moindre carre \n=======================")
    # print(moindreCarre(res[0], res[1], km, price))
    # bonus()

# ========= BONUS =========
def bonus():
    result = []
    result = parse()
    km = result[0]
    price = result[1]
    plt.plot(km, price, 'ro')
    plt.ylabel('Price (euro))')
    plt.xlabel('Kilometrage (km)')
    plt.plot([0.0, 250000.0], [8500.0, 3500.0], 'b--', lw=2) # Red straight line
    # plt.savefig('StraightLine.png')
    plt.show()

def bonus_secondprgoram(a, b):
    result = []
    result = parse()
    km = result[0]
    price = result[1]
    plt.plot(km, price, 'gx')
    plt.ylabel('Price (euro)')
    plt.xlabel('Kilometrage (km)')
    plt.plot([a * x + b for x in range(250000)])
    # plt.savefig('StraightLine.png')
    plt.show()

def bonus2():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.rand(100, 1)

    X_b = np.c_[np.ones((100, 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(theta_best)

    plt.figure(figsize=(10, 10))
    predictions = plt.plot(X_new, y_predict, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.figlegend(predictions, "pred", 'upper center')
    plt.show()

def bonus_estimation(a, b):
    plt.plot([a * x + b for x in range(250000)])
    plt.show()

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
print("- pas d'argument : 2nd prgm\nsinon\n- rajouter mettre un kilometrage")
if len(sys.argv) == 2:
    try:
        km = float(sys.argv[1])
        premierPgrm(km)
    except ValueError:
        print("That's not an int!")
elif len(sys.argv) == 1:
    res = []
    res = parse()
    print(yaxb(res[0], res[1]))
