# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:10:54 2024

@author: Caleb Erickson
"""

#import package needed to compute execution time
import time
#import matplotlib to enable plotting histogram
import matplotlib.pyplot as plt
#lab2 solution
#First generate a series of numbers, for which to compute square root
#constants to determine start, stopPlusOneStep, step value for range function
START_VAL = 5
#NOTE: the STOP_VAL is one STEP_VAL beyond the last number generated
STOP_VAL = 5005
STEP_VAL = START_VAL
#first create an empty list:
nums = []
#generate all multiples of 5, from 5 to 5000, inclusive
for i in range(START_VAL, STOP_VAL, STEP_VAL):
    #append each new number to the nums list
    nums.append(i)

    
#create an empty list for the square roots computed using bisection search// bisecion code
bis_ans = []
#create an empty list for the number of iterations required using bisection method
bis_num_it = []
#set the desired value of epsilon
epsilon = 0.01
#implement bisection search to compute square root of all the numbers in the nums list
#determined the start time at which bisection search started
bis_start_time = time.time()
#use for loop to iterate through all numbers, computing for each the square root using bisection search
for i in range(len(nums)):
    #begin assigning elements to the list keeping track of number of iterations required
    #each element begins with zero, which will be succesively incremented by one with each iteration
    bis_num_it.append(0)
    #initialize variables needed for bisection method
    low = 0.0
    high = max(1.0, nums[i])
    ans = (high + low)/2.0
    #iterate the bisetion search method until the estimated square root is within wpsilon of the correct value
    while abs(ans**2 - nums[i]) >= epsilon:
        #increment by one this element of the bisNumIt list
        bis_num_it[i] += 1
        #determine in which half of the search space the correct answer is located
        if ans**2 < nums[i]:
            low = ans
        else:
            high = ans
        #now that the correct area of search space is located, update the estimate of the square root (i.e. ans)
        ans = (high + low)/2.0
    #when the diff between the estimate and the correct answer is less than epsilon, append this to the bis_ans list
    bis_ans.append(ans)
#determine the time at which the bisection search was completed
bis_end_time = time.time()

#compute the execution time required to execure bisection search
bis_exe_time = bis_end_time - bis_start_time
#determine ave number of iterations
sum_iterations = 0
for i in range(len(bis_num_it)):
    sum_iterations += bis_num_it[i]
    
ave_iterations = sum_iterations / len(bis_num_it)


#Newton code
nr_ans = []
nr_num_it = []
epsilon = 0.01
k = nums[0]
nr_start_time = time.time()

for i in range(len(nums)):
    
    nr_num_it.append(0)
    
    guess = k/2
    while abs(guess**2 - k) >= epsilon:
        nr_num_it[i] += 1
        guess = guess - (((guess**2) - k)/(2*guess))
        ans1 = guess
        k = nums[i]
    nr_ans.append(ans1)
    
nr_end_time = time.time()

nr_exe_time = nr_end_time - nr_start_time

sum1_iterations = 0
for i in range(len(nr_num_it)):
    sum1_iterations +=nr_num_it[i]
    
ave1_iterations = sum1_iterations / len(nr_num_it)


#Secant code
sec_ans = []
sec_num_it = []
epsilon = 0.01
j = nums[0]
sec_start_time = time.time()

for i in range(len(nums)):
    
    sec_num_it.append(0)
    
    x0 = nums[i]/2
    x1 = nums[i]
    
    while (abs(x1 - x0))>=epsilon:
        sec_num_it[i] += 1
        
        #compute f0 & f1
        f0 = x0 ** 2 - nums[i]
        f1 = x1 ** 2 - nums[i]
        
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)  # Secant method formula
        x0 = x1
        x1 = x_next
        
    sec_ans.append(x1)
    
sec_end_time = time.time()
sec_exe_time = sec_end_time - sec_start_time

# Compute average number of iterations
sum2_iterations = sum(sec_num_it)
ave2_iterations = sum2_iterations / len(nums)


#display results for Bisection
print('Use two different algorithms to compute square roots of multiples of', STEP_VAL, 'in the following range:')
print('[', START_VAL, ',', (STOP_VAL - STEP_VAL), ']')
print('***************************')
print("BISECTION SEARCH ALGORITHM:")
print('Max num iterations required:', max(bis_num_it))
print('Min num iterations required:', min(bis_num_it))
#average number of iterations, rounded to four places
print('Ave number of iterations required:', round(ave_iterations, 4))
#execution time rounded to six places
print('Bisection search execution time:', f'{bis_exe_time:.6f}')
print('***************************')
#display results for Newton
print('***************************')
print('NEWTON-RAPHSON ALGORITHM:')
print('Max num iterations required:', max(nr_num_it))
print('Min num iterations required:', min(nr_num_it))
#average number of iterations, rounded to four places
print('Ave number of iterations required:', round(ave1_iterations, 4))
#execution time rounded to six places
print('Newton-Ralphson execution time:', f'{nr_exe_time:.6f}')
print('***************************')
#display results for Secant
print('***************************')
print('SECANT SEARCH ALGORITHM:')
print('Max num iterations required:', max(sec_num_it))
print('Min num iterations required:', min(sec_num_it))
#average number of iterations, rounded to four places
print('Ave number of iterations required:', round(ave2_iterations, 4))
#execution time rounded to six places
print('Secant execution time:', f'{sec_exe_time:.6f}')
print('***************************')


#plot histogram to compare efficiency of bisection search and Newton-Raphson algorithms
#set up number of 'bins'
bins = 10
#single histogram to plot both lists
plt.hist(nr_num_it, bins, facecolor = 'blue', label = 'Num Newton iterations')
plt.hist(sec_num_it, bins, facecolor = 'red', label = 'Num Secant iterations')
plt.hist(bis_num_it, bins, facecolor = 'green', label = 'Num Bisection iterations')
#format features of the histogram
plt.legend(loc = 'upper right')
plt.xlabel('Number of iterations required')
plt.ylabel('Num occurences of each iteration value')
plt.title('Number of Iterations: Bisection vs Newton vs Secant')