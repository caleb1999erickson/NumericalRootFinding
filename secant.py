#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
secant method code
Created on 

@author: Caleb Erickson
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math
  
'''
define secant method

secant: This finds the root of a given function via the secant method
=====================================
input f : this is the function
      x0 : this is a value a little to the left of the root
      x1 : this is a value a little to the right of the root
      tol: (optional argument) this is a small positive value that tells us when to stop, as we are within tol of our actual root
      MaxIter: (optional argument) This is the maxIteration that we are willing to go to
output returns x1, which is the approximated root via the Secant Method
'''
def secant(f, x0, x1, tol = 1e-5, MaxIter = 50):
    
    if tol <= 0:
        raise ValueError(f"tolerance = {tol} must be positive")
    if MaxIter <= 0:
        raise ValueError(f"Maximum number of iterations = {MaxIter} must be positive")
    if isinstance(MaxIter, int) == False:
        raise ValueError(f"MaxIter = {MaxIter} must be an integer")
    if (abs(x1 - x0) <= 0):
        raise ValueError(f"x0 = {x0} and x1 = {x1} must be different")
    
    f0 = f(x0)
    f1 = f(x1)
    i=2
    
    while (((5*abs(x1 - x0) + abs(f1))>tol) and (i < MaxIter+2)):
        if (f1 -f0)!= 0:
            x = x1 - f1*(x1-x0)/(f1-f0)
        else:
            raise RuntimeError("Division by zero in secant method")
        
        fx = f(x)
        x0 = x1
        x1 = x
        f0 = f1
        f1 = fx
        i = i+1
        
        if i>=(MaxIter+2):
            print("Secant method did not converge")
            
    return x1, i-2



'''
define secant_diagnostic method

secant_diagnostic: This creates a list of closer and closer approximations to the actual root of the function.
=====================================
input f : this is the function
      x0 : this is the initial x
      x1 : this is the first x (after initial x)
      tol: (optional argument) this is a small positive value that tells us when to stop, as we are within tol of our actual root
      MaxIter: (optional argument) This is the maxIteration that we are willing to go to
output returns xa[0:i], returns a list of approximated roots, getting closer to the actual root
'''
def secant_diagnostic(f, x0, x1, tol = 1e-5, MaxIter = 50):
    
    if tol <= 0:
        raise ValueError(f"tolerance = {tol} must be positive")
    if MaxIter <= 0:
        raise ValueError(f"Maximum number of iterations = {MaxIter} must be positive")
    if isinstance(MaxIter, int) == False:
        raise ValueError(f"MaxIter = {MaxIter} must be an integer")
    if abs(x1-x0) <= 0:
        raise ValueError(f"x0 = {x0} and x1 = {x1} must be different")
    
    xa = np.zeros(MaxIter + 2)
    xa[0] = x0
    xa[1] = x1
    f0 = f(x0)
    f1 = f(x1)
    i=2
    
    while (((5*abs(x1 - x0) + abs(f1))>tol) and (i < MaxIter+2)):
        if (f1 -f0)!= 0:
            x = x1 - f1*(x1-x0)/(f1-f0)
            xa[i] = x
        else:
            raise RuntimeError("Division by zero in secant method")
        
        fx = f(x)
        x0 = x1
        x1 = x
        f0 = f1
        f1 = fx
        i = i+1
        
        if i>=(MaxIter+2):
            print("Secant method did not converge")
            
    return xa[0:i], i-2


def bisect(f,a,b, eps):
    if a>= b:
        raise ValueError(f"We need a<b but we have a={a} and b={b}.")
    if eps<=0 :
        raise ValueError(f"Parameter eps={eps} should be positive.")
    fa = f(a)
    fb = f(b)
    if fa*fb>0:
        raise RuntimeError("Function does not change sign on interval.")
    n = math.ceil(math.log2(b-a) - math.log2(eps))
    for i in range(n):
        c = a+0.5*(b-a)
        fc = f(c)
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            if fa*fc > 0:
                a = c
                fa = fc
            else:
                alpha = c
                return alpha
    return c, i




tol = 1e-8

#Test 1
def tf1(x):
    return np.cos( math.sqrt(x) )


xr1 = (np.arccos(0))**2
x01 = 2
x11 = 4

x = secant(tf1, x01, x11, tol)
assert abs(x[0] - xr1)<tol



#Test 2
def tf2(x):
    return 3*x**2 + 2*x - 5

xr2 = 1
x02 = .8
x12 = 1.2

x = secant(tf2, x02, x12, tol)
assert abs(x[0] - xr2)<tol



#Test 3
def tf3(x):
    return (np.exp(2*x) - 2)

xr3 = np.log(2)/2
x03 = 0
x13 = 1

x = secant(tf3, x03, x13, tol)
assert abs(x[0] - xr3)<tol





#Testing f(x)=x^{25}-1
def tf4(x):
    return (x**25)-1

xr4 = 1
x04 = .9
x14 = 1.1

x = secant(tf4, x04, x14, tol)
assert abs(x[0] - xr4)<tol


#testing number of iterations needed to converge (with Secant)
eps = 10e-8
x1 = secant_diagnostic(tf4,x04,x14,tol)



#return results to user
print("\n\n****************")
print(f"To use secant to approximate the zero of f(x) = x^25-1 to {tol} error")
print(f"we need {x1[1]} iterations with x0={x04} and x1={x14}.")
print("****************\n\n")




#defining a function g(x) with an iterative method
n = 10
def g(x):
    a = 1
    
    # Iterate to compute a_n
    for i in range(n):
        a = a + ((x * math.cos(a) + x)/n)
    
    return a

#Test the function with x = 1
result = g(1)
print("g(1) =", result)


# Plot g(x)
plt.plot(np.linspace(-10, 10, 100), [g(x) for x in np.linspace(-10, 10, 100)])
plt.title('Plot of g(x)')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.grid(True)
plt.show()
plt.close()



def f(x):
    return g(x) - 2  # Define your function f(x) = g(x) - 2
   
# Initial guess
x0 = 0
x1 = 1

# Call secant method
alpha = secant(f, x0, x1, tol = 10e-12)

print("Approximate root:", alpha[0])


eps = 10e-6
alpha1 = secant_diagnostic(f, x0, x1, tol = eps)

print(alpha1[1])

k = alpha1[1]



#beta1 = bisect(f, x0, x1, eps)
#first number is the root, second number is the iteration loop number
#print(beta1)
#j = beta1[1]
bis_iterations = math.ceil((math.log(x1 - x0) - math.log(eps)) / math.log(2))



eps1 = 10**-6
tol = 10e-6
print("\n\n****************")
print(f"alpha = {alpha[0]}")
print(f"It took {k} iterations to get to 10e-6 accuracy with secant.")
print(f"This will require {k+2} function calls to get 10e-6 accuracy with secant.")
print(f"It will take {bis_iterations} iterations to get 10e-6 accuracy with bisect.") #for some reason eps kept printing as 10e-5, even though I reset it before print statement
print(f"It will take {2*bis_iterations} function calls to get 10e-6 accuracy with bisect.")
print("****************\n\n")





# Call secant method
approximated_root, iterations = secant(f, x0, x1, tol=10e-12)

# Test against true root to verify the number of iterations required
real_root = alpha[0]
while abs(approximated_root - real_root) > tol:
    approximated_root, _ = secant(f, x0, x1, tol=10e-12)
    iterations += 1

print("Making sure 4 is the actual number required (for secant iterations):", iterations-2)











if __name__ == '__main__':
    def f(x):
        return x*x - 4

    
    tol = -0.01
    try: 
        secant(f,0,3,tol)
    except ValueError as e:
        msg = f"tolerance = {tol} must be positive"
        assert str(e) == msg
        
    try: 
        secant_diagnostic(f,0,3,tol)
    except ValueError as e:
        msg = f"tolerance = {tol} must be positive"
        assert str(e) == msg

        
    tol = 0
    try: 
        secant(f,0,3,tol)
    except ValueError as e:
        msg = f"tolerance = {tol} must be positive"
        assert str(e) == msg
     
    try: 
        secant_diagnostic(f,0,3,tol)
    except ValueError as e:
        msg = f"tolerance = {tol} must be positive"
        assert str(e) == msg
    
    tol = 1e-10
    N = 11.2
    try:
        secant(f,0,3,tol,N)
    except ValueError as e:
        msg = f"MaxIter = {N} must be an integer"
        assert str(e) == msg
    
    try:
        secant_diagnostic(f,0,3,tol,N)
    except ValueError as e:
        msg = f"MaxIter = {N} must be an integer"
        assert str(e) == msg
           
    N = -100
    try:
        secant(f,0,3,tol,N)
    except ValueError as e:
        msg = f"Maximum number of iterations = {N} must be positive"
        assert str(e)==msg
    
    try:
        secant_diagnostic(f,0,3,tol,N)
    except ValueError as e:
        msg = f"Maximum number of iterations = {N} must be positive"
        assert str(e)==msg  
    
        
    
    #f(x) = x^2 -4
    #so use x = +1 has f(x) = -3
    #and x = -1 has f(x) = -3
    #This will generate the division by zero in secant.
    
    try:
        secant(f,-1,1)
    except RuntimeError as e:
        assert str(e) == "Division by zero in secant method"
    
    try:
        secant_diagnostic(f,-1,1)
    except RuntimeError as e:
        assert str(e) == "Division by zero in secant method"
        
    def f(x):
        return x*x - 0.5
    N = 20
    tol=1e-8
    try:
        secant(f,3,3,tol,N)
    except ValueError as e:
        msg = f"x0 = {3} and x1 = {3} must be different"
        assert str(e) == msg
    
    try:
        secant_diagnostic(f,3,3,tol,N)
    except ValueError as e:
        msg = f"x0 = {3} and x1 = {3} must be different"
        assert str(e) == msg       
    
    x = secant(f,0,3,tol,N)
    assert abs(x[0]-math.sqrt(0.5))<tol
    
    x1 = secant_diagnostic(f,0,3,tol,N)
    assert(abs(x1[0][-1]-math.sqrt(0.5))<tol)

    assert abs(x1[0][-1]-x[0])==0
       
    x = secant(math.sin, 1, 4,tol)
    assert abs(x[0]-np.pi)<tol
    
    x1 = secant_diagnostic(math.sin, 1, 4,tol)
    assert abs(x1[0][-1]-np.pi)<tol  
    assert x[0] == x1[0][-1]
    
    def h(x):
        return x**2 - 4*x - 4
    #root  2+2*math.sqrt(2) is approx 4.8
    x = secant(h,2,6,tol)
    assert abs(x[0]-(2+2*math.sqrt(2)))<tol
    
    
    x1 = secant_diagnostic(h,2,6,tol)
    assert abs(x1[0][-1]-(2+2*math.sqrt(2)))<tol 
    assert x[0] == x1[0][-1]
    

    #root 2-2*math.sqrt(2) is approx -0.83
    x = secant(h,0,-2,tol)
    assert abs(x[0]-(2-2*math.sqrt(2)))<tol
       
    x1 = secant_diagnostic(h,0,-2,tol)
    assert abs(x1[0][-1]-(2-2*math.sqrt(2)))<tol   
    assert x[0] == x1[0][-1]
       
    def g(x):
        return 2-np.exp(x)
 
    
    x = secant(g,0,1,tol)
    assert(abs(x[0] - math.log(2))<tol)
    
    x1 = secant_diagnostic(g,0,1,tol)
    assert(abs(x1[0][-1]-math.log(2))<tol)
    assert((abs(x1[0][-1]-x[0]))==0)
    
    