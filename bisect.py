#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bisection method code

@author: Caleb Erickson
"""


import math as math
import numpy as np

'''
define bisection method

bisect: finds the roots of f on the interval [a,b] via bisection method
=====================================
input f : continuous function on interval [a,b]
      a : the lower bound
      b : the upper bound
      eps: 10**-8
output returns an array ... which approximates the zero

'''
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
    return c
    

#tests to check the bisection algorithm
if __name__ == '__main__':
    eps = 10**-8
    b = 1.2
    a = 0.2
    try:
        c=bisect(math.sin, b, a, eps)
    except ValueError as e:
        assert str(e) == "We need a<b but we have a=1.2 and b=0.2." #error is a>=b
    try:
        c = bisect(math.sin, a, b, eps)
    except RuntimeError as e:
        assert str(e) == "Function does not change sign on interval." #error if the function does not have roots
    try:
        c = bisect(math.sin, a, b, -eps)
    except ValueError as e:
        s = str(f"Parameter eps={-eps} should be positive.") #error if eps is negative
        assert str(e) == s
    
    #testing bisection to find x_0 = pi for function sin(x)
    b = 4.
    a = 1.
    c = bisect(math.sin, a, b, eps)
    assert abs(c-np.pi)<eps
    
    
    
    #test given, finding the roots of x^2 - 4x - 4
    def f(x):
        return x**2 - 4*x - 4
    b = 6.
    a = 1.
    c = bisect(f, a, b, eps)
    assert abs(c-(2+2*math.sqrt(2)))<eps
    b = 0
    a = -2
    c = bisect(f, a, b, eps)
    assert abs(c-(2-2*math.sqrt(2)))<eps
    
    
    #test 1 // testing bisection to find the roots of x^2 + 6x - 16
    def f(x):
        return x**2 + 6 * x - 16
    b = 5.
    a = 0.
    c = bisect(f, a, b, eps)
    assert abs(c - 2) < eps #-2 is a root
    
    b = -5.
    a = -10.
    c = bisect(f, a, b, eps)
    assert abs(c + 8) < eps #8 is a root
    
    
    
    #test 2 // testing bisection to find x_0 = 3*pi/2 for function cos(x)
    def f(x):
        return np.cos(x)
    b = 2*np.pi
    a = np.pi
    x_0 = 3*np.pi/2
    c = bisect(f, a, b, eps)
    assert abs(c - x_0) < eps
    
    
    
    #test 3 // testing bisection to find x_0 = ln(11/7) for function: 1-(7 / 11)*math.exp(x)
    def f(x):
        return 1 - (7 / 11) * math.exp(x)
    b = 2
    a = -2
    x_0 = 0.4519851237
    c = bisect(f, a, b, eps)
    assert abs(c - x_0) < eps

    
    
    
    