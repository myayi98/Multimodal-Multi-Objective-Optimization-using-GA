# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Title: Effect of RTS on Multi-modal Optimization
Version: 1.0
Created on Sat Mar 15 2022
@author: Mahesh
"""
import numpy as np
import math
import matplotlib.pyplot as plt

#Function to calculate fitness = xsin(10πx)+1
#def calc_fitness(x):
#    return x * math.sin(10 * math.pi * x) + 1

def calc_fitness(x):
    return x*math.sin(10*math.pi*x)+1

#Function for performing base-10 encoding
def gene_encode(x, chromosome_size):
    charlist = [char for char in str(x)]
    if charlist[0] == '-':
        del charlist[0]
        del charlist[1]
        returnlist = [int(char) for char in charlist]
        returnlist.insert(0, -1)
    else:
        del charlist[1]
        returnlist = [int(char) for char in charlist]
        returnlist.insert(0, 1)
    while len(returnlist) != chromosome_size:
        returnlist.append(0)
    return returnlist

#Function to decode base-10 genes
def gene_decode(gene):
    try:
        if len(gene) > 0:
            val = 0
            for i in range(1,len(gene)):
                val += gene[i]/10**(i-1)
            val *= gene[0]
            return val
    except:
        return gene

class Pop:
    pop_count = 0
    pop_list = []
    def __init__(self,value):
        self.id = Pop.pop_count
        self.value = value
        self.decoded_value = gene_decode(self.value)
        self.fitness = calc_fitness(self.decoded_value)
        Pop.pop_count += 1
        Pop.pop_list.append(self)
    def __repr__(self):
        return 'id='+str(self.id)+' :: value='+str(gene_decode(self.value))+' :: pr='+str(self.pareto_fitness)

#Function to generate uniform population distribution across the domain
def gen_pop(pop_size,domain,chromosome_size):
    for i in range(pop_size):
        pop = round(np.random.uniform(domain[0], domain[1]), chromosome_size-2)
        Pop(gene_encode(pop,chromosome_size))

def ts(pop_list,parent_size,ksize):
    parent_list = []
    while len(parent_list) <= parent_size:
        # print('while 1')
        kpop = {}
        while len(kpop) <= ksize:
            # print('while 2')
            # print(len(kpop))
            # print(ksize)
            randint = np.random.randint(0,len(pop_list))
            key = pop_list[randint].fitness
            if not key in kpop:
                kpop[key] = pop_list[randint]
            else:
                kpop[key+np.random.uniform(0,0.0000000001)] = pop_list[randint]
        parent_list.append(kpop[max(kpop.keys())])
    return parent_list

#Function for mutating base-10 genes
def mutation(pop):
    for gene in range(len(pop)):
        if gene == 0:
            if np.random.uniform(0,1) >= 0.5:
                pop[0] *= -1
        else:
            pop[gene] = int(10*np.random.uniform(0,1))
    return pop

def rts_selection(pop):
    distance = 99
    similar_pop = None
    decoded_pop = gene_decode(pop)
    for i in Pop.pop_list:
        gap = abs(decoded_pop - i.decoded_value)
        if gap <= distance:
            similar_pop = i
            distance = gap
    if similar_pop.fitness < calc_fitness(decoded_pop):
        del Pop.pop_list[Pop.pop_list.index(similar_pop)]
        Pop(pop)

# GA function
def run_ga(initial_pop,domain,pc,pm,generations,parent_size):
    for i in range(generations):
        parent_list = ts(initial_pop,parent_size,int(parent_size/2))
        a = parent_list[np.random.randint(0,len(parent_list))].value
        b = parent_list[np.random.randint(0,len(parent_list))].value
        while a == b:
            b = parent_list[np.random.randint(0,len(parent_list))].value
        pc_randvar = np.random.random()
        crossover_point = np.random.randint(0,len(a)+2) #crossover point is chosen randomly for every parent pair selection
        if pc_randvar >= pc:
            child1 = a[:crossover_point] + b[crossover_point:] 
            child2 = b[:crossover_point] + a[crossover_point:]
        else:
            child1 = a[:]
            child2 = b[:]
        if np.random.random() >= pm:
            child1 = mutation(child1)
        if np.random.random() >= pm:
            child2 = mutation(child2)
        if not (domain[0]<=gene_decode(child1)<domain[1] and domain[0]<=gene_decode(child2)<domain[1]):
            i -= 1
            continue

        rts_selection(child1)
        rts_selection(child2)

if __name__ == "__main__":
    print('miniproject4')
    pop_size = 50
    parent_size = 6
    domain = [-.5,1]
    pc = 0.4
    pm = .4
    generations = 500*20
    chromosome_size = 8
    gen_pop(pop_size,domain,chromosome_size)
    run_ga(Pop.pop_list,domain,pc,pm,generations,parent_size)
    plt.plot([i for i in np.arange(domain[0],domain[1],0.01)],[calc_fitness(i) for i in np.arange(domain[0],domain[1],0.01)],label='Objective fn = xsin(10πx)+1')
    plt.scatter([i.decoded_value for i in Pop.pop_list],[i.fitness for i in Pop.pop_list],label='Candidate Solutions')
    plt.title('Multi-modal optimization using RTS')
    plt.xlabel('Genotype Space')
    plt.ylabel('Phenotype Space')
    plt.legend()
    plt.show()