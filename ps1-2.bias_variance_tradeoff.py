#homework 1-2: bias variance tradeoff

import random
import numpy as np
import math
import scipy.integrate as integrate

random.seed()

print "Choose hypothesis set:"
print "1. h(x) = b"
print "2. h(x) = ax"
print "3. h(x) = ax+b"
print "4. h(x) = ax**2"
print "5. h(x) = ax**2+b"
hypothesis = int(raw_input("Hypothesis set: "))
runs = int(raw_input("Number of runs: "))
testpoints = int(raw_input("Number of testing points: "))
#Ein = np.array([0]*runs, dtype = np.float)
Eout = np.array([0]*runs, dtype = np.float)
avalues = np.array([0]*runs, dtype = np.float)
bvalues = np.array([0]*runs, dtype = np.float)

def mse(x, a_avg, b_avg, hypothesis):
    if hypothesis == 1:
        a = 0
        b = b_avg
        power = 1
    elif hypothesis == 2:
        a = a_avg
        b = 0
        power = 1
    elif hypothesis == 3:
        a = a_avg
        b = b_avg
        power = 1
    elif hypothesis == 4:
        a = a_avg
        b = 0
        power = 2
    elif hypothesis == 5:
        a = a_avg
        b = b_avg
        power = 2
    else:
        print "Invalid hypothesis!"
        exit(0)
    dif = a*(x**power) + b - math.sin(math.pi*x)
    return 0.5*(dif**2)

xa = np.array([0]*runs, dtype = np.float)
xb = np.array([0]*runs, dtype = np.float)
ya = np.array([0]*runs, dtype = np.float)
yb = np.array([0]*runs, dtype = np.float)    
    
for i in range(runs):
    #the target function is f(x) = sin(pi x) on [-1,1]
    
    #choose xa, xb, the training points; load y
    xa[i] = random.uniform(-1,1)
    xb[i] = random.uniform(-1,1)
    ya[i] = math.sin(xa[i]*math.pi)
    yb[i] = math.sin(xb[i]*math.pi)
    
    #hypothesis minimizes square error 
    
    #Let h(x)=b; if g=h then b = (ya + yb)/2
    if hypothesis == 1:
        bvalues[i] = (ya[i]+yb[i])/2
    
    #Let h(x)=ax; if g=h then a = (xaya + xbyb)/(xa^2 + xb^2)
    elif hypothesis == 2:
        num = xa[i]*ya[i] + xb[i]*yb[i]
        den = xa[i]**2 + xb[i]**2
        avalues[i] = num/den
        
    #Let h(x)=ax+b; if g=h then a = (ya-yb)/(xa-xb) and b = ya - a*xa = yb - a*xb
    elif hypothesis == 3:
        avalues[i] = (ya[i]-yb[i])/(xa[i]-xb[i])
        fbv = [ya[i]-avalues[i]*xa[i],yb[i]-avalues[i]*xb[i]]
        bvalues[i] = (fbv[0]+fbv[1])/2
        
    #Let h(x)=ax**2; if g=h then a = (ya-yb)/(xa-xb) and b = ya - a*xa = yb - a*xb
    elif hypothesis == 4:
        num = xa[i]*xa[i]*ya[i] + xb[i]*xb[i]*yb[i]
        den = xa[i]**4 + xb[i]**4
        avalues[i] = num/den
    
    #Let h(x)=ax**2+b; if g=h then a = (ya-yb)/(xa**2-xb**2) and b = ya - a*xa**2 = yb - a*xb**2
    elif hypothesis == 5:
        avalues[i] = (ya[i]-yb[i])/(xa[i]**2-xb[i]**2)
        fbv = [ya[i]-avalues[i]*(xa[i]**2),yb[i]-avalues[i]*(xb[i]**2)]
        bvalues[i] = (fbv[0]+fbv[1])/2
    
    #choose test_x_n, the testing points; load test_y
    test_x = np.array([0]*testpoints, dtype=float)
    test_fy = np.array([0]*testpoints, dtype=float)
    test_gy = np.array([0]*testpoints, dtype=float)
    E_gy = np.array([0]*testpoints, dtype=float)
    for j in range(testpoints):
        test_x[j] = random.uniform(-1,1)
        test_fy[j] = math.sin(math.pi*test_x[j])
        if hypothesis == 1:
            test_gy[j] = bvalues[i]
        elif hypothesis == 2:
            test_gy[j] = test_x[j]*avalues[i]
        elif hypothesis == 3:
            test_gy[j] = test_x[j]*avalues[i] + bvalues[i]
        elif hypothesis == 4:
            test_gy[j] = test_x[j]*test_x[j]*avalues[i]
        elif hypothesis == 5:
            test_gy[j] = test_x[j]*test_x[j]*avalues[i] + bvalues[i]
        E_gy[j] = (test_fy[j] - test_gy[j])**2
    
    Eout[i] = np.mean(E_gy)
    
'''
print np.average(xa)
print np.average(xb)
print np.average(ya)
print np.average(yb)
'''    
a_avg = np.mean(avalues)     
b_avg = np.mean(bvalues)    
#if h(x) = b (hypothesis 1), mean g(x) = b_avg
if hypothesis == 1:
    print "mean g(x) = b_avg; b_avg is %.2f" % b_avg

#if h(x) = ax (hypothesis 2), mean g(x) = a_avg*x    
elif hypothesis == 2:
    print "mean g(x) = a_avg * x; a_avg is %.2f" % a_avg

#if h(x) = ax+b (hypothesis 3), mean g(x) = a_avg*x + b_avg   
elif hypothesis == 3:
    print "mean g(x) = a_avg*x + b_avg; a_avg is %.2f and b_avg is %.2f" % (a_avg, b_avg)
    
else:
    print "a_avg is %.2f and b_avg is %.2f" % (a_avg, b_avg)
    
#bias is integral from -1 to 1 of (g(x)-sin(pi x))**2 dx times 1/2 (uniform distribution)
bias = integrate.quad(lambda x: mse(x, a_avg, b_avg, hypothesis), -1, 1)
print "Bias is %.2f" % bias[0]

#variance is integral from -1 to 1 of E(D)[(h(x)-g(x))**2] dx times 1/2
if hypothesis == 1:
    bdifsq = (bvalues - b_avg)**2
    e_bdifsq = np.mean(bdifsq)
    var = e_bdifsq
elif hypothesis == 3:
    adif = (avalues - a_avg)
    bdif = (bvalues - b_avg)
    # so (h(x)-g(x))**2 = (adif*x+bdif)**2 = adif**2 * x**2 + 2*adif*bdif*x + bdif**2
    coef1 = np.mean(adif**2)
    coef2 = np.mean(2*adif*bdif)
    coef3 = np.mean(bdif**2)
    integr = integrate.quad(lambda x: coef1*x*x+coef2*x+coef3, -1, 1)
    var = 0.5*integr[0]
else:
    var = np.mean(Eout).item() - bias[0]
print "Variance is %.2f" % var
print "Out of sample error is %.2f" % np.mean(Eout).item()
