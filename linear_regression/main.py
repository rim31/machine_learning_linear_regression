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

def normalise(km, price):
    minPrice = min(price)
    maxPrice = max(price)
    minKm = min(km)
    maxKm = max(km)
    for i in range(len(km)):
        price[i] = (price[i] - minPrice) / (maxPrice - minPrice)
        km[i] = (km[i] - minKm) / (maxKm - minKm)
        # print(km, price)
    return (km, price)

def denormalize(km, price, a, b, kilometre):
    a = np.float64(a)
    b = np.float64(b)
    maxPrice = max(price)
    minPrice = min(price)
    maxKm = max(km)
    minKm = min(km)
    x = (float(kilometre) - minKm) / (maxKm - minKm)
    y = b + a * x
    price = minPrice + y * (maxPrice - minPrice)
    # print(price)



def calculGradient(km, price):
    iterations = 1000
    learningRate = 0.1
    a_current = 0.0
    b_current = 0.0
    m = float(len(km))
    moindrecarre = []
    for i in range(iterations):
        sumDiff0 = 0
        sumDiff1 = 0
        for j in range(len(km)):
            sumDiff0 += b_current + a_current * km[j] - price[j]
            sumDiff1 += (b_current + a_current * km[j] - price[j]) * km[j]
        b_current = b_current - learningRate * sumDiff0 / m
        a_current = a_current - learningRate * sumDiff1 / m
        if (i % 200) == 0:
            bonus_normalise(km, price, a_current, b_current)
            # print(moindreCarre(a_current, b_current, km, price))
            moindrecarre.append(moindreCarre(a_current, b_current, km, price))
            # print(a_current, b_current)
    bonus_normalise(km, price, a_current, b_current)
    print(denormalize(km, price, a_current, b_current, 123))
    bonus_moindrecarre(moindrecarre)
    return(a_current, b_current)


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
    secondPrgm(a, b, km, price, 0.1)
    return(a, b)

def secondPrgm(a, b, km, price, learningRate):
    # print("Methode moindre carre \n=======================")
    # print(moindreCarre(a, b, km, price))
    bonus_secondprgram(a, b)
    # print(a, b)
    res = []
    res = normalise(km, price)
    calculGradient(res[0], res[1])
    # print(res)
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

def bonus_secondprgram(a, b):
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

def bonus_moindrecarre(moindrecarre):
    plt.plot(moindrecarre, 'bo')
    plt.xlabel('moindre carre (km)')
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

def bonus_normalise(km, price, a, b):
    plt.plot(km, price, 'gx')
    plt.ylabel('Price (euro)')
    plt.xlabel('Kilometrage (km)')
    plt.plot([a * x + b for x in range(2)])
    # plt.savefig('StraightLine.png')
    plt.show()

def premierPgrm(km):
    t0 = np.loadtxt("theta0.txt", unpack='true')
    t1 = np.loadtxt("theta1.txt", unpack='true')
    # print(t0, t1)
    print("penser a verifier les coef t0.txt et t1.txt")
    print("your car worth ")
    price = t0 + (t1 * km)
    print(firstprgm(km, t0, t1))


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
