from __future__ import division

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from arch import arch_model
from scipy.stats import norm
from matplotlib import cm
from scipy import interpolate as interpolate
import math as mth
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from matplotlib.backends.backend_pdf import PdfPages

# Do logging to stdout
def log(*ss):
    for s in ss: print(s)

# Do some plots as pdf files
def doplots(basefn):
    i=0
    for T in expirs:
        i+=1
        pp=PdfPages(basefn+'-'+str(i)+'.pdf')
        plt.figure(i,figsize=(8.0,10.0))
        plt.subplot(311)
        plt.title('Slice '+str(i)+': Expiration T='+str(T))
        plt.xlabel('Log-Strike')
        plt.ylabel('Option price')
        t=data.loc[data['Expiration']==T,'LogStrike']
        c=data.loc[data['Expiration']==T,'Mid_Matrix']
        tt=grid[T]
        w=[straightSVI(k,chi.loc[T,'m1'],chi.loc[T,'m2'],chi.loc[T,'q1'],chi.loc[T,'q2'],chi.loc[T,'c']) for k in grid[T]]
        m=[BlackScholes("C",S0,K,r,sig,T,q) for K,sig in zip(grid.index,np.sqrt(w/T))]
        plt.plot(t,c,'bo',tt,m,'k')
        plt.subplot(312)
        plt.xlabel('Log-Strike')
        plt.ylabel('Implied volatility')
        iv=data.loc[data['Expiration']==T,'IV']
        cv=np.sqrt(w/T)
        plt.plot(t,iv,'bo',tt,cv,'k')
        plt.subplot(313)
        plt.xlabel('Log-Strike')
        plt.ylabel('Risk neutral density')
        p=[RND(k,chi.loc[T,'m1'],chi.loc[T,'m2'],chi.loc[T,'q1'],chi.loc[T,'q2'],chi.loc[T,'c']) for k in grid[T]]
        plt.plot(tt,p,'k')
        plt.savefig(pp,format='pdf')
        pp.close()

# Black Scholes formula
def BlackScholes(type, S0, K, r, sigma, T, q):
    def d1(S0, K, r, sigma, T, q):
        return (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    def d2(S0, K, r, sigma, T, q):
        return (np.log(S0 / K) + (r - q - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    if type == "C":
        return S0 * np.exp(- q * T) * norm.cdf(d1(S0, K, r, sigma, T, q)) - K * np.exp(-r * T) * norm.cdf(
            d2(S0, K, r, sigma, T, q))
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2(S0, K, r, sigma, T, q)) - S0 * np.exp(-q * T) * norm.cdf(
            -d1(S0, K, r, sigma, T, q))

# Raw SVI parametrization
def rawSVI(k,a,b,rho,m,sigma):
    return a+b*(rho*(k-m)+np.sqrt((k-m)**2+sigma**2))

# Hyperbola asymptotes parametrisation
def straightSVI(x,m1,m2,q1,q2,c):
    return ((m1+m2)*x+q1+q2+np.sqrt(((m1+m2)*x+q1+q2)**2-4*(m1*m2*x**2+(m1*q2+m2*q1)*x+q1*q2-c)))/2
def straightSVIp(x,m1,m2,q1,q2,c):
    H=np.sqrt(((m1+m2)*x+q1+q2)**2-4*(m1*m2*x**2+(m1*q2+m2*q1)*x+q1*q2-c))
    return ((m1+m2)+((m1+m2)*((m1+m2)*x+q1+q2)-4*m1*m2*x-2*(m1*q2+m2*q1))/H)/2
def straightSVIpp(x,m1,m2,q1,q2,c):
    H=np.sqrt(((m1+m2)*x+q1+q2)**2-4*(m1*m2*x**2+(m1*q2+m2*q1)*x+q1*q2-c))
    A=(2*(m1+m2)**2-8*m1*m2)/H
    B=(2*(m1+m2)*((m1+m2)*x+q1+q2)-8*m1*m2*x-4*(m1*q2+m2*q1))**2/H**3/2
    return (A-B)/4

# Alternative parametrisation including the parabola
def stdSVI(x,a0,a1,a2,a3,a4):
    return (-a1*x-a3+np.sqrt((a1*x+a3)**2-4*a0*(a2*x**2+x+a4)))/(2*a0)

# Obtain asymptotic parameters from alternative parametrization
def std2straight(a):
    m1=-a[1]/a[0]/2.-np.sqrt((a[1]/a[0])**2/4.-a[2]/a[0])
    m2=-a[1]/a[0]/2.+np.sqrt((a[1]/a[0])**2/4.-a[2]/a[0])
    q1=(1+m1*a[3])/a[0]/(m2-m1)
    q2=(1+m2*a[3])/a[0]/(m1-m2)
    c =q1*q2-a[4]/a[0]
    return [m1,m2,q1,q2,c]

# Obtain rawSVI parameters from asymptotic parametrization
def straight2raw(chi):
    a=(chi[0]*chi[3]-chi[1]*chi[2])/(chi[0]-chi[1])
    b=abs(chi[0]-chi[1])/2.
    rho=(chi[0]+chi[1])/abs(chi[0]-chi[1])
    m=-(chi[2]-chi[3])/(chi[0]-chi[1])
    sigma=np.sqrt(4*chi[4])/abs(chi[0]-chi[1])
    return [a,b,rho,m,sigma]

# Calculate risk neutral density wrt logstrike
def RND(k,m1,m2,q1,q2,c):
    w=straightSVI(k,m1,m2,q1,q2,c)
    wp=straightSVIp(k,m1,m2,q1,q2,c)
    wpp=straightSVIpp(k,m1,m2,q1,q2,c)
    g=(1.-k*wp/(2.*w))**2-wp**2/4.*(1./w+1./4.)+wpp/2.
    return g/np.sqrt(2*np.pi*w)*np.exp(-0.5*((-k-w/2.)**2/w))

# Define some constants
S0      = 179.83             # asset price
q       = 0.026              # dividends
r       = 0.005              # interest rate
sig0    = 0.2                # initial volatility guess
lvol    = 0.05               # lower volatility acceptance limit
uvol    = 3.00               # upper volatility acceptance limit
bpen    = 128                # initial butterfly penalty factor
cpen    = 128                # initial calendar penalty factor
blim    = 0.001              # target butterfly arbitrage bound
clim    = 0.001              # target calendar arbitrage bound

# Read raw data
log('Reading raw data ...')
data = pd.read_csv('data.csv') #raw data from csv
num=pd.DataFrame(index=sorted(set(data['Expiration'])))
num['Raw']=[len(data.loc[data['Expiration']==T]) for T in sorted(set(data['Expiration']))]

# Add column LogStrike to data
def logstrike(K,T): return np.log(K/S0*np.exp(-(r-q)*T))
data['LogStrike']=[logstrike(K,T) for K,T in zip(data['Strike'],data['Expiration'])]

# Helper function to solve for implied volatility
def bsaux(sigma, type, S0, K, r, T, q, C): return BlackScholes(type, S0, K, r, sigma, T, q)-C
log('Calculating implied volatilities ...')
data['IV']=[sp.optimize.bisect(bsaux,-1,100,args=("C",S0,K,r,T,q,C),xtol=1e-3) for K,T,C in zip(data['Strike'],data['Expiration'],data['Mid_Matrix'])]

# Clean raw data wrt an implied volatility bound and report number of records
log('Cleaning data to ensure '+str(lvol)+' <= IV <= '+str(uvol)+' ...')
data=data.loc[(data['IV']>lvol) & (data['IV']<uvol), :]
num['Clean']=[len(data.loc[data['Expiration']==T]) for T in sorted(set(data['Expiration']))]
log('Number of records in raw and cleaned dataset:',num)

# Prepare grid on which to check presence of arbitrage
expirs = sorted(set(data['Expiration']))
strikes = sorted(set(data['Strike']))
grid=pd.DataFrame(index=strikes)
for T in expirs: grid[T]=[logstrike(K,T) for K in strikes]

# Variable to store parameter vectors chi
chi=pd.DataFrame(index=expirs,columns=['m1','m2','q1','q2','c'])

# Residuals function for fitting implied volatility
def residSVI(chi,T):
    w=[straightSVI(k,chi[0],chi[1],chi[2],chi[3],chi[4]) for k in data.loc[data['Expiration']==T,'LogStrike']]
    return data.loc[data['Expiration']==T,'IV']-np.sqrt(w/T)

# Function to obtain initial parameter vector for fit
def chi0(T):
    # Split data in five intervals and calculate mean x and mean y
    kmin=np.min(data.loc[data['Expiration']==T,'LogStrike'])
    kmax=np.max(data.loc[data['Expiration']==T,'LogStrike'])
    klo=[kmin+i*(kmax-kmin)/5. for i in range(5)]
    kup=[kmin+(i+1)*(kmax-kmin)/5. for i in range(5)]
    xm=np.array([np.mean(data.loc[(data['Expiration']==T) & (l<=data['LogStrike']) & (data['LogStrike']<=u),'LogStrike']) for l,u in zip(klo,kup)])
    ym=np.array([np.mean(T*data.loc[(data['Expiration']==T) & (l<=data['LogStrike']) & (data['LogStrike']<=u),'IV']**2) for l,u in zip(klo,kup)])

    # Determine quadratic form through these five average points
    un=np.array([1 for l in klo])
    A=np.matrix([ym*ym,ym*xm,xm*xm,ym,un]).T
    a=np.linalg.solve(A,-xm)

    # If it's already a hyperbola, we have our initial guess
    if 4*a[0]*a[2]<a[1]**2: return std2straight(a)

    # Otherwise, flip to approximating hyperbola and do a least squares fit to the five points
    a[0]=-a[0]
    def residHyp(chi):
        return np.array([straightSVI(x,chi[0],chi[1],chi[2],chi[3],chi[4]) for x in xm])-ym
    ap=sp.optimize.leastsq(residHyp,std2straight(a))
    return ap[0]

# Fit implied volatilities directly to obtain first guess on parameter vectors
i=0
log('Calculating first guess on parameters ...')
for T in expirs:
    i+=1
    log('Fitting implied volatility on slice '+str(i)+', T='+str(T)+' ...')
    chi.loc[T,:]=chi0(T)
    chi.loc[T,:]=sp.optimize.leastsq(residSVI,list(chi.loc[T,:]),args=(T))[0]
    log('Got parameters:',chi.loc[T,:])
log('Summary of initial guess for parameters:',chi)

# Function to quantify calendar arbitrage between two slices T1 > T2 on grid
def calendar(chi1,T1,chi2,T2):
    if T2==0 or T1<=T2: return 0
    w1=[straightSVI(k,chi1[0],chi1[1],chi1[2],chi1[3],chi1[4]) for k in grid[T1]]
    w2=[straightSVI(k,chi2[0],chi2[1],chi2[2],chi2[3],chi2[4]) for k in grid[T2]]
    return sum([np.maximum(0,x2-x1) for x1,x2 in zip(w1,w2)])

# Function to quantify butterfly arbitrage in a slice on grid
def butterfly(chi,T):
    w=np.array([straightSVI(k,chi[0],chi[1],chi[2],chi[3],chi[4]) for k in grid[T]])
    wp=np.array([straightSVIp(k,chi[0],chi[1],chi[2],chi[3],chi[4]) for k in grid[T]])
    wpp=np.array([straightSVIpp(k,chi[0],chi[1],chi[2],chi[3],chi[4]) for k in grid[T]])
    g=(1.-(grid[T]*wp)/(2.*w))**2-wp**2/4.*(1./w+1./4.)+wpp/2.
    return sum([np.maximum(0,-x) for x in g])

# Residuals function for fitting option prices with penalties on arbitrage
def residuals(chiT,T,Tp):
    w=[straightSVI(k,chiT[0],chiT[1],chiT[2],chiT[3],chiT[4]) for k in data.loc[data['Expiration']==T,'LogStrike']]
    bs=[BlackScholes("C",S0,K,r,sig,T,q) for K,sig in zip(data.loc[data['Expiration']==T,'Strike'],np.sqrt(w/T))]
    calarbT=calendar(chiT,T,chi.loc[Tp,:],Tp) if Tp else 0
    butarbT=butterfly(chiT,T)
    e=data.loc[data['Expiration']==T,'Mid_Matrix']-bs
    return e+(np.sqrt(sum(e)**2+(cpen*calarbT+bpen*butarbT)**2*len(e))-sum(e))/len(e)

# Reduce arbitrage by fitting option prices with penalties on calendar and butterfly arbitrage
maxbutarb=float("Inf")
maxcalarb=float("Inf")
while maxbutarb>blim or maxcalarb>clim:
    log('Butterfly penalty factor: '+str(bpen))
    log('Calendar penalty factor: '+str(cpen))
    j=0
    Tp=0
    maxbutarb=0
    maxcalarb=0
    for T in expirs:
        j+=1
        log('Fitting mid prices on slice '+str(j)+', T='+str(T)+' ...')
        chi.loc[T,:]=sp.optimize.leastsq(residuals,list(chi.loc[T,:]),args=(T,Tp))[0]
        log('Got parameters:',chi.loc[T,:])
        butarb=butterfly(chi.loc[T,:],T)
        log('Butterfly penalty for slice is '+str(bpen*butarb))
        calarb=calendar(chi.loc[T,:],T,chi.loc[Tp,:],Tp) if Tp else 0
        log('Calendar penalty for slice is '+str(cpen*calarb))
        maxbutarb=np.maximum(maxbutarb,butarb)
        maxcalarb=np.maximum(maxcalarb,calarb)
        Tp=T
    if maxbutarb>clim: bpen*=2
    if maxcalarb>clim: cpen*=2

log('Maximum remaining butterfly arbitrage is '+str(maxbutarb))
log('Maximum remaining calendar arbitrage is '+str(maxcalarb))
log('Summary of final parameters:',chi)

#
# Report raw parameters and draw plots with final fit
#
raw=pd.DataFrame(index=expirs,columns=['a','b','rho','m','sigma'])
for T in expirs: raw.loc[T,:]=straight2raw(chi.loc[T,:])
log('Final raw SVI parameters:',raw)
doplots('SVI104')

#def discriminant(a,b,c,d,e):
#    return 256*a**3*e**3-192*a**2*b*d*e**2-128*a**2*c**2*e**2 +144*a**2*c*d**2*e-27*a**2*d**4\
#           + 144*a*b**2*c*e**2 - 6*a*b**2*d**2*e -80*a*b*c**2*d*e+18*a*b*c*d**3+16*a*c**4*e\
#           -4*a*c**3*d**2-27*b**4*e**2+18*b**3*c*d*e-4*b**3*d**3-4*b**2*c**3*e+b**2*c**2*d**2
