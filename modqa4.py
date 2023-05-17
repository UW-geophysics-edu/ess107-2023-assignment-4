import numpy as np
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
from scipy.signal import detrend


def alpha(T,alpha1,alpha2):
    T1 = 273.15 - 5
    T2 = 273.15 + 5
    
    if T<T1:
        return alpha1
    elif (T<T2) and (T>T1):
        return alpha1  +(alpha2-alpha1)*(T-T1)/(T2-T1)
    else:
        return alpha2

def budyko(y, t, S,epsilon,sigma,C,alpha1,alpha2):
    dydt =  (S*(1-alpha(y,alpha1,alpha2)) - epsilon*sigma*y**4) / C
    return dydt

def global_temperature_model(S,albedo_ice=0.9,albedo_noice=0.1):
    y0 = 270
    t = np.linspace(0, 5000, 10001)
    sigma = 5.670e-8
    epsilon = 0.5
    C = 700 # Heat capacity of the atmosphere
    output=[]
    
    for thisS in S:
        sol = odeint(budyko, y0, t, args=(thisS,epsilon,sigma,C,albedo_ice,albedo_noice) )
        output.append(sol[-1, 0])
        
    return np.array(output)

def easy_fft(data,dt=1):
    data = np.array(data)
    N = len(data)
    ft = np.abs(fft(detrend(data)))
    ft_scaled = 2.0/N * np.abs(ft[0:N//2])
    f = fftfreq(N, dt)[:N//2]
    period = 1/f
    return ft_scaled,period