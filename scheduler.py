#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator

#Reads data and stores each user's appliance usage in an array
def read_excel():
    Input_excel = pd.read_excel("./Input.xlsx", sheet_name = 'User & Task ID')
    Task_ID = list(Input_excel['User & Task ID'])
    Ready_Time = list(Input_excel['Ready Time'])
    Deadline = list(Input_excel['Deadline'])
    Maximum_per_hour = list(Input_excel['Maximum scheduled energy per hour'])
    Energy_Demand = list(Input_excel['Energy Demand'])
    
    tasks = []
    
    for i in range(50):
        each_task = []
        each_task.append(Ready_Time[i])
        each_task.append(Deadline[i])
        each_task.append(Maximum_per_hour[i])
        each_task.append(Energy_Demand[i])
        tasks.append(each_task)
        
#Store user information in a dictionary, with keys user1 to user5. 
#The value is a list that stores the user's appliance usage in the order: 
#"User & Task ID, Ready Time, Deadline, Maximum scheduled energy per hour, Energy Demand.
    user_tasks_dic = {}

    user1_task = []
    user2_task = []
    user3_task = []
    user4_task = []
    user5_task = []
    for i in range(10):
        user1_task.append(tasks[i])
    for i in range(10,20):
        user2_task.append(tasks[i])
    for i in range(20,30):
        user3_task.append(tasks[i])
    for i in range(30,40):
        user4_task.append(tasks[i])
    for i in range(40,50):
        user5_task.append(tasks[i])

    user_tasks_dic['user1'] = user1_task
    user_tasks_dic['user2'] = user2_task
    user_tasks_dic['user3'] = user3_task
    user_tasks_dic['user4'] = user4_task
    user_tasks_dic['user5'] = user5_task
    
    return user_tasks_dic


"""Read the abnormal predictive guideline price curve predicted from classification.py, 
#which have been saved in the file 'TestingResults.txt'"""
def read_testing():
    columns = list(map(str, range(24))) + ['label']
    testing_results = pd.read_csv('dataset/TestingResults.txt', sep=',', names=columns)
    abnormal_prices = testing_results.loc[testing_results['label'] == 0]
    abnormal_prices.pop('label')
    return abnormal_prices

#Define linear programming algorithms
#Objective function: min = hourly price factor * hourly electricity consumption
# Constraint: 0 <= Hourly consumption <= Maximum scheduled energy per hour
# Ready Time <= Electricity consumption time <= Deadline
# Sum of hourly electricity consumption = Energy Demand 
def lpsolve(tasks,abnormal_prices):
    cost = []
    price_each_hour = []
    scheduled_hour = []
#Creat model 
    model = LpProblem(name="scheduling-problem", sense=LpMinimize)

#Constraints
    for i in range(tasks[0],tasks[1]+1):
        each_price = abnormal_prices[i]
        price_each_hour.append(each_price)
        x = LpVariable(name='Userhour_' + str(i), lowBound=0, upBound=tasks[2], cat='Continuous')
        scheduled_hour.append(x)
        
# Objective Function
    for i in range( 0 ,tasks[1]+1 - tasks[0]):
        cost += price_each_hour[i] * scheduled_hour[i]
    model += cost
    model += lpSum(scheduled_hour) == tasks[3]

    model.solve()
    return model


"""Call the minimum cost calculation function lpsolve() for 10 tasks per user, expressed as a one-dimensional array
This function is used to compute the linear programming based energy scheduling solution according to the abnormal predictive guideline
price curve using lpsolve. 
The size of the input data is (tasks,abnormal_prices), where tasks is a list including Usage per task. 
Each entry represents how much energy is scheduled for a task per hour. 'Abnormal_prices' represents the abnormal predictive guideline price curve""" 
def single_user_schedule(tasks, ab):
    schedule_single_user = [0]*24
    for i in range(10):
        model = lpsolve(tasks[i],ab)
        for variable in model.variables():
            schedule_single_user[int(variable.name.split("_")[1])] += variable.varValue
    return schedule_single_user

#Call the function defined in the previous section to read the data
user_tasks_dic = read_excel()
abnormal_prices = read_testing()
os.makedirs("./charts/")  
#Iterate through each abnormal guide price and calculate the minimum consumption
for z in range(len(abnormal_prices)):
    ab = abnormal_prices.iloc[z]
    user_schedule = {}
    total_energy = [0]*24
    for j in range(1,6):
        exec("user = user_tasks_dic['user%s']"%(j))
        schedule = single_user_schedule(user,ab)
        exec("user_schedule['user%s'] = schedule"%(j))
    for i in range(1,6):
        exec("user = user_schedule['user%s']"%(i))
        for j in range(24):
            if user[j] != 0:
                total_energy[j] += user[j]
                
#Generates a bar chart of the the hourly energy usage of this community             
    columns = list(map(str, range(25)))
    total_energy.append(0)
    plt.figure(dpi = 200, figsize = (6,2))
    plt.bar(range(25), total_energy, align = 'edge', color=[1.0, 0.5, 0.25], tick_label = columns)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=5, rotation=0)
    #Sets the font size of the x and y axis scale values; rotation specifies the horizontal alignment of the scale text.
    plt.xticks(fontsize=5)    
    #Set the font size of the x-axis scale values
    plt.yticks(fontsize=5)
    #Sets the font size of the x and y axis scale values; rotation specifies the horizontal alignment of the scale text.

    plt.xlabel('Hour(h)',fontsize=5,rotation=0)
    plt.ylabel('Energy Usage(kw)',fontsize=5)
    #Save output files locally
    exec("plt.savefig('./charts/Abnormal_Price%s.jpg')"%(z))
    plt.close()

