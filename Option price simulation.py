#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# # Monte Carlo Simulation - European Options

# In[2]:


# Discrete stock price process under Q-measure
def Stock(sigma, r, T, n, dt, S0):
    S = np.zeros(n)
    S[0] = S0
    for t in range(n-1):
        S[t+1] = S[t] * np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1))
    return S

sigma = 0.25                   #volatility
r = 0.01                       #riskfree-rate
T = 0.05                        #years
n = int(365 * T)             #nr. of price calculations
dt = T/n                       #time increment between prices
S0 = 100                        #initial stock price
K  = 99                       #Strike price (of the option contracts)
R = 1000                       #nr. of replications

S = np.zeros([n,R])
for rows in range(R):
    S[:,rows] = Stock(sigma, r, T, n, dt, S0)

fig1 = plt.figure(figsize=[16,6])
plt.plot(S);
plt.xlabel("Day");
plt.ylabel("Price");
plt.xlim([0,n-1]);
plt.title("Stock Price Monte Carlo Simulation (R=%i)" %R);


# In[3]:


# European Call Option Price
X = np.clip(S[-1,:] - K,0,None)
C = np.exp(-r*T)*np.mean(X);

# European Put Option Price
X = np.clip(K - S[-1,:],0,None)
P = np.exp(-r*T)*np.mean(X);


# In[4]:


# BS call and put price
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

# BS European Call Option Price
C_BS = S0 * norm.cdf(d1) - np.exp(-r*T) * K * norm.cdf(d2) 

# BS European Put Option Price
P_BS = -S0 * norm.cdf(-d1) + np.exp(-r*T) * K * norm.cdf(-d2) 


# In[5]:


print("Monte Carlo European Call Option Price at Strike", K , "USD =", "and", n, "days till maturity =", np.round(C,2), "USD")
print("Black Scholes European Call Option Price at Strike", K , "USD =", "and", n, "days till maturity =", np.round(C_BS,2), "USD")
print("Monte Carlo European Put Option Price at Strike", K , "USD =", "and", n, "days till maturity =", np.round(P,2), "USD")
print("Black Scholes European Put Option Price at Strike", K , "USD =", "and", n, "days till maturity =", np.round(P_BS,2), "USD")


# # Binomial Options Pricing Model - Cox, Ross, & Rubinstein (CRR) Method for American Options

# In[6]:


# nr. of steps in tree
m = 18
# consider call=0 or put=1
optiontype = 1


# In[7]:


# plot empty tree lettice
fig2 = plt.figure(figsize=[17, m+5])
ax = fig2.add_subplot(111)
plt.title("Binomial Tree American Put Option Prices with K=%i" %K)
loc_x = []
loc_y = []
for i in range(m):
    x = [1, 0, 1]
    for j in range(i):
        x.append(0)
        x.append(1)
    x = np.array(x) + i
    y = np.arange(-(i+1), i+2)[::-1]
    plt.plot(x, y, 'ko-')
    if i == 0:
        loc_x[0:m+2*i] = x
        loc_y[0:m+2*i] = y
    else:
        loc_x = np.append(loc_x,x)
        loc_y = np.append(loc_y,y)

plt.ylim(-1.3*m,1.3*m+0.3);
ax = plt.gca()
ax.axes.yaxis.set_visible(False)

# get stock prices at each node 
u = np.exp(sigma * np.sqrt(dt))      #up factor from the condition that the variance of the log of the price is sigma^2 * t
d = 1/u                              #down factor        
loc_S = S0*u**loc_y 

#plot stock prices
for z in range((m+1)**2-1):
    ax.text(loc_x[z], loc_y[z]+0.2, "%1.2f" %loc_S[z], ha="center")
    
#calculate probability q
delta_t = T/m
q = (np.exp(r*delta_t) - d) / (u - d)

#Get option prices
M = np.zeros([m+1,m+1])

#end nodes
counter1 = 0;
for i in range((m+1)**2-1-(2*m+1),(m+1)**2-1,2):
    if optiontype == 1:
        M[counter1,-1] = np.clip(K - loc_S[i],0,None)
    if optiontype == 0:
        M[counter1,-1] = np.clip(loc_S[i] - K,0,None)
    counter1 += 1

#work backwards to option price at t = 0    
for h in range(m):
    counter2 = 0    
    for j in range(m-h):
        M[counter2,-(h+2)] = np.exp(-r*delta_t) * (q * M[counter2,-(h+2)+1] + (1-q)*M[counter2+1,-(h+2)+1]) 
        counter2 += 1
        
#Get stock prices in matrix        
Mat_S = np.zeros([m+1,m+1])
Mat_S[:,0] = np.arange(0,m+1,1)
for c in range(1,m+1,1):
    Mat_S[:,c] = Mat_S[:,c-1] - 2 
Mat_S = 100*u**Mat_S
Mat_S = np.tril(Mat_S)
Mat_S = np.transpose(Mat_S)

#plot option prices at each node
for k in range(1,m+2,1):
    for z in range(k):
        if k==1:
            ax.text(k-1, loc_y[z+1]+0.8, "%1.2f" %M[z,k-1], ha="center", color='blue')        
        else:
            if K - Mat_S[z, k-1] >= M[z,k-1]:
                ax.text(k-1, loc_y[2*z+(k-1)**2-1]+0.8, "%1.2f" %M[z,k-1], ha="center", color='red')
            else:
                ax.text(k-1, loc_y[2*z+(k-1)**2-1]+0.8, "%1.2f" %M[z,k-1], ha="center", color='blue')


# In[ ]:





# In[ ]:




