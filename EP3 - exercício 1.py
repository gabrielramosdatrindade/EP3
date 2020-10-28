# Terceiro Programa - Integração numérica
# Gabriel Ramos - 10737460


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

################# Precisao simples

def f(x):
    return (np.float32(7)-np.float32(5)*np.float32(x)**4)

x0,xf = 0,1 # Limites da integral

iteração = np.zeros(25)
solução_numerica = np.zeros(25)

for p in range (1,26):
    N = 2**p
    h = np.float32(1/N)
    
    x = np.float32(x0) + np.arange(N+1 , dtype = np.float32)*np.float32(h)
    coef = np.ones(N + 1, dtype = np.float32)
    coef[0] = np.float32(0.5)
    coef[-1] = np.float32(0.5)
    S_final = h*(np.sum(coef*f(x), dtype = np.float32))
    
    print()
    print("Iteração", p)
    print("O valor analítico é", S_final)
    print("O erro é", np.abs(S_final - 6))
    
    iteração[p-1] = p
    solução_numerica[p-1] = S_final
    S_final = 0

#################### Precisao dupla
    
def g(x):
    return (7-5*(x)**4)

x0,xf = 0,1 # Limites da integral

iteração2 = np.zeros(25)
solução_numerica2 = np.zeros(25)

for p in range (1,26):
    N = 2**p
    h = (1/N)
    
    x = (x0) + np.arange(N+1)*(h)
    coef = np.ones(N + 1)
    coef[0] = (0.5)
    coef[-1] = (0.5)
    S_final = h*(np.sum(coef*g(x)))
    
    print()
    print("Iteração", p)
    print("O valor analítico é", S_final)
    print("O erro é", np.abs(S_final - 6))
    
    iteração2[p-1] = p
    solução_numerica2[p-1] = S_final
    S_final = 0

df1 = pd.DataFrame({'p' : iteração, 'erro' :np.abs(solução_numerica - 6)})
df2 = pd.DataFrame({'p': iteração2, 'erro' : np.abs(solução_numerica2 - 6)})

dg1 = df1[(df1['erro'] != 0) & (df1['p'] > 15)]
dg2 = df2[df2['erro'] != 0]
plt.scatter(dg1['p'], np.log10(dg1['erro']), marker = '^')
plt.scatter(dg2['p'], np.log10(dg2['erro']), marker = 'o')

################# Ajustes

def ajuste(x,a,b):
    return (a*x + b)

popt, pcov = curve_fit(ajuste, dg1['p'], np.log10(dg1['erro']))
################# Preparando arquivo csv
    
#iteração_final = np.array([iteração]).T
#solução_numerica_final = np.array([solução_numerica]).T
#
#doc_final = np.concatenate((iteração_final, solução_numerica_final), axis = 1)
#
#np.savetxt("precisaodupla.csv", doc_final, delimiter = ",")



