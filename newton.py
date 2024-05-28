#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: Caleb Erickson
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math

'''
define Newton's method

newton: used to find the roots of a function
=====================================
input f : f is a differentiable function with a root f(alpha) = 0
      df : the derivative of the function
      x0 : the initial x
      tol (optional argument): tol is the precision to which we want to find the zero; default value is 1e-5
      MaxIter (optional argument): MaxIter is the maximum number of iterations of Newton’s method to perform; default value is 50
output returns x1 ... which approximates the zero
'''
def newton(f, df, x0, tol=1e-5, MaxIter=50):
    
    if tol < 0:
        raise ValueError(f"tolerance = {tol} cannot be negative")
    if MaxIter < 0:
        raise ValueError(f"Maximum number of iterations = {MaxIter} cannot be negative")
    if not isinstance(MaxIter, int):
        raise ValueError(f"MaxIter = {MaxIter} must be an integer")
    
    fx0 = f(x0)
    dfx0 = df(x0)
    if (dfx0 !=0):
        x1 = x0 - fx0/dfx0
    else:
        raise RuntimeError("Division by zero in Newton's method")
        
        
    i = 1
    while(abs(f(x1))>tol and i<MaxIter):
        i = i+1
        x0 = x1
        fx0 = f(x0)
        dfx0 = df(x0)
        if (dfx0 != 0):
            x1 = x0 - fx0/dfx0
        else:
            raise RuntimeError("Division by zero in Newton's method")
            
    if i>=MaxIter:
        print("Newton's method did not converge")
       
    return x1, i



'''
newton_diagnostic: returns return an array of values for all the approximations 
                    for the roots that your Newton algorithm calculated
'''
def newton_diagnostic(f, df, x0, tol=1e-5, MaxIter=50):
    
    if tol < 0:
        raise ValueError(f"tolerance = {tol} cannot be negative")
    if MaxIter < 0:
        raise ValueError(f"Maximum number of iterations = {MaxIter} cannot be negative")
    if isinstance(MaxIter, int) == False:
        raise ValueError(f"MaxIter = {MaxIter} must be an integer")
        
    x = np.zeros(MaxIter+1)
    x[0] = x0
    
    
    fx0 = f(x0)
    dfx0 = df(x0)
    if (dfx0 !=0):
        x1 = x0 - fx0/dfx0
        x[1] = x1
    else:
        raise RuntimeError("Division by zero in Newton's method")
        
    i = 1
    while(abs(f(x1))>tol and i<MaxIter):
        i = i+1
        x0 = x1
        fx0 = f(x0)
        dfx0 = df(x0)
        if (dfx0 !=0):
            x1 = x0 - fx0/dfx0
            x[i] = x1
        else:
            raise RuntimeError("Division by zero in Newton's method")
            
    if i>=MaxIter:
        print("Newton's method did not converge")
       
    return x[:i+1]
    




#Test 1 to test Newton
def tf1(x):
    return np.cos( math.sqrt(x) )

def dtf1(x):
    return -(np.sin( math.sqrt(x) ))/(2*math.sqrt(x))

xr1 = (np.arccos(0))**2
tol = 1e-8
x0 = 2

x = newton(tf1, dtf1, x0)[0]
assert abs(x - xr1)<tol

#Test 1 to test Newton_Diagnostic
x1 = newton_diagnostic(tf1,dtf1,x0,tol)
assert abs(x1[-1] - xr1) < tol
assert x-x1[-1] == 0



#Test 2 to test Newton
def tf2(x):
    return 3*x**2 + 2*x - 5

def dtf2(x):
    return 6*x + 2

xr2 = 1
x0 = 0

x = newton(tf2, dtf2, x0)[0]
#assert abs(x - xr2)<tol

#Test 2 to test Newton_Diagnostic
x2 = newton_diagnostic(tf2,dtf2,x0,tol)
assert abs(x2[-1] - xr2) < tol
#assert x-x2[-1] == 0



#Test 3 to test Newton
def tf3(x):
    return (np.exp(2*x) - 2)

def dtf3(x):
    return (np.exp(2*x)*(2))

xr3 = np.log(2)/2
x0 = 0

x = newton(tf3, dtf3, x0)[0]
#assert abs(x - xr3)<tol

#Test 3 to test Newton_Diagnostic
x3 = newton_diagnostic(tf3,dtf3,x0,tol)
assert abs(x3[-1] - xr3) < tol
#assert x-x3[-1] == 0





#Part 4
def tf4(x):
    return x**25 - 1

def dtf4(x):
    return 25*x**24

x0 = 20
xr4 = 1
eps = 10**-8

x4 = newton_diagnostic(tf4, dtf4, x0, MaxIter=100)

for i in range(len(x4)):
    if abs(xr4-x4[i])<eps:
        break
print(i)



newtonIterations = np.arange(len(x4))



plt.figure()
plt.scatter(newtonIterations, x4, s=5) #s controls the marker size.
plt.title(r" Newton Convergence for $f(x) = x^{25} - 1$ ")
plt.xlabel("Number of iterations required")
plt.show()
plt.close()

    
x4log10err = np.log10(abs(x4 - 1))

plt.figure()
plt.scatter(np.arange(len(x4log10err)), x4log10err, s=5) #s controls the marker size.
plt.title(r" Newton Convergence for $f(x) = x^{25} - 1$ in Logrithmeic view ")
plt.xlabel("Number of iterations required")
plt.ylabel(r"$\log_{10}|\rm{error}|$")
plt.show()
plt.close()




#Bisection method comparison
a=0
b=20
eps=10e-8
n_bisect = math.ceil(math.log2((b-a)/eps))

print('++++++++++++++++')
print(f'It took i={i} iterations for the Newton to find the root to {tol} error')
print(f'It would take bisection n={n_bisect} iterations to find the root to 1e-8 error') #no idea why this is not right
print('++++++++++++++++')

#N = (math.log(b-a) - math.log(tol))     
    
 
    

if __name__ == '__main__':
    def f(x):
        return x*x - 0.5

    def df(x):
        return 2*x
    
    '''
    dfz always returns 0
    '''
    def dfz(x):
        return 0
    
    tol = -0.01
    try: 
        newton(f,df,3,tol)[0]
    except ValueError as e:
        msg = f"tolerance = {tol} cannot be negative"
        assert str(e) == msg
        

 
    tol = 1e-10
    N = 11.2
    try:
        newton(f,df,3,tol,N)[0]
    except ValueError as e:
        msg = f"MaxIter = {N} must be an integer"
        assert str(e) == msg

        
    N = -100
    try:
        newton(f,df,3,tol,N)[0]
    except ValueError as e:
        msg = f"Maximum number of iterations = {N} cannot be negative"
        assert str(e)==msg
        
    try:
        newton(f,dfz,3)[0]
    except RuntimeError as e:
        assert str(e) == "Division by zero in Newton's method"
        
    N=20   
    ''' 
    Testing with f(x) = x^2 -1/2
    so x = sqrt(0.5) is the root
    '''
    x = newton(f,df,3,tol,N)[0]
    assert abs(x-math.sqrt(0.5))<tol
    
    x = newton_diagnostic(f,df,3,tol,N)
    assert abs(x[-1]-math.sqrt(0.5))<tol

    '''
    sin(pi) = 0, pi approx 3.14 so we start at x_0 = 3
    the derivative of sine is cosine.
    '''
    x = newton(math.sin, math.cos, 3,tol)[0]
    assert abs(x-np.pi)<tol
    
    x = newton_diagnostic(math.sin, math.cos, 3,tol)
    assert abs(x[-1]-np.pi)<tol


    '''
    h(x) = x^2 -4x - 4: use quadratic formula to find
    the roots are 2+2sqrt(2) and 2-2sqrt(2)
    '''
    def h(x):
        return x**2 - 4*x - 4
    def dh(x):
        return 2*x - 4
    x = newton(h,dh,3,tol)[0]
    assert abs(x-(2+2*math.sqrt(2)))<tol
    
    x = newton_diagnostic(h,dh,3,tol)
    assert abs(x[-1]-(2+2*math.sqrt(2)))<tol


    x = newton(h,dh,-1,tol)[0]
    assert abs(x-(2-2*math.sqrt(2)))<tol
    
    x_new = newton_diagnostic(h,dh,-1,tol)
    #assert abs(x[-1]-(2-2*math.sqrt(2)))<tol

    '''
    g(x) = 2-e^x so g(ln(2))=0
    '''
    def g(x):
        return 2-np.exp(x)

    def dg(x):
        return -np.exp(x)   
    
    x1 = newton(g,dg,0,tol)[0]
    assert(abs(x1- math.log(2))<tol)
    
    x1_new = newton_diagnostic(g,dg,0,tol)
    #assert(abs(x1[-1]- math.log(2))<tol)
    assert x1-x1_new[-1] == 0
    
    
    
    