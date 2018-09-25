
# coding: utf-8

# In[1]:

#importing modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import math
from scipy import signal
from scipy import ndimage
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')

from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# In[2]:

#reading csv files

exotrain=pd.read_csv("ExoTrain.csv")
exotest = pd.read_csv('Final Test.csv')


# In[3]:

#checking the dimensions-

print(exotrain.shape)
print(exotest.shape)
exotest.head()


# In[4]:

#BLS Algorithm

def bls(t, x, qmi, qma, fmin, df, nf, nb):
    
    n = len(t); rn = len(x)
    #! use try
    if n != rn:
        print "Different size of array, t and x"
        return 0

    rn = float(rn) # float of n

    minbin = 5
    nbmax = 2000
    if nb > nbmax:
        print "Error: NB > NBMAX!"
        return 0

    tot = t[-1] - t[0] # total time span

    if fmin < 1.0/tot:
        print "Error: fmin < 1/T"
        return 0

    # parameters in binning (after folding)
    kmi = int(qmi*nb) # nb is number of bin -> a single period
    if kmi < 1: 
        kmi = 1
    kma = int(qma*nb) + 1
    kkmi = rn*qmi # to check the bin size
    if kkmi < minbin: 
        kkmi = minbin

    # For the extension of arrays (edge effect: transit happen at the edge of data set)
    nb1 = nb + 1
    nbkma = nb + kma
        
    # Data centering
    t1 = t[0]
    u = t - t1
    s = np.mean(x) # ! Modified
    v = x - s

    bpow = 0.0
    p = np.zeros(nf)
    # Start period search
    for jf in range(nf):
        f0 = fmin + df*jf # iteration in frequency not period
        p0 = 1.0/f0

        # Compute folded time series with p0 period
        ibi = np.zeros(nbkma)
        y = np.zeros(nbkma)
        for i in range(n):
            ph = u[i]*f0
            ph = ph - int(ph)
            j = int(nb*ph) # data to a bin 
            ibi[j] = ibi[j] + 1 # number of data in a bin
            y[j] = y[j] + v[i] # sum of light in a bin
        
        # Extend the arrays  ibi()  and  y() beyond nb by wrapping
        for j in range(nb1, nbkma):
            jnb = j - nb
            ibi[j] = ibi[jnb]
            y[j] = y[jnb]

        # Compute BLS statictics for this trial period
        power = 0.0

        for i in range(nb): # shift the test period
            s = 0.0
            k = 0
            kk = 0
            nb2 = i + kma
            # change the size of test period (from kmi to kma)
            for j in range(i, nb2): 
                k = k + 1
                kk = kk + ibi[j]
                s = s + y[j]
                if k < kmi: continue # only calculate SR for test period > kmi
                if kk < kkmi: continue # 
                rn1 = float(kk)
                powo = s*s/(rn1*(rn - rn1))
                if powo > power: # save maximum SR in a test period
                    power = powo # SR value
                    jn1 = i # 
                    jn2 = j
                    rn3 = rn1
                    s3 = s

        power = math.sqrt(power)
        p[jf] = power

        if power > bpow:
            bpow = power # Save the absolute maximum of SR
            in1 = jn1
            in2 = jn2
            qtran = rn3/rn
            high = -s3/(rn - rn3)
            low = s3/rn3
            depth = high - low
            bper = p0
    
    # ! add
    sde = (bpow - np.mean(p))/np.std(p) # signal detection efficiency

    return bpow, in1, in2, qtran, depth, bper, sde, high, low


# In[5]:

#preparing data for bls algorithm

data=exotest
labels= data['LABEL']
data=data.drop(['LABEL'],axis=1)
data_norm=pd.DataFrame(data)


# In[6]:

print(data.shape)
data.head()


# In[7]:

stats.describe(data_norm)


# In[8]:

#extracting the features using bls algorithm and storing them in res

fluxes=[0 for i in range(2000)]
result=pd.DataFrame()
res=[0 for i in range(2000)]
t = np.linspace(0,1000,3197)
qmi = 0.01
qma = 0.1
fmin = 0.3 
df = 0.001 
nf = 1000
nb = 200
for i in range(0,1):
    fluxes=data.iloc[i,:]
    res[i] = bls(t, fluxes, qmi, qma, fmin, df, nf, nb)


# In[9]:

for i in [59, 499, 999, 1499]:
    flux = exotrain[exotrain.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]
    time = np.arange(len(flux)) * (36.0/(60.0*24)) # time in units of days
    plt.figure(figsize=(12,4))
    plt.title('Flux of star {} with no exoplanets'.format(i+1))
    plt.ylabel('Flux')
    plt.xlabel('Time in days')
    plt.plot(time, flux)


# In[10]:

# Obtaining flux for several stars with exoplanets from the train data:

for i in [2,4,9]:
    flux = exotrain[exotrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
    time = np.arange(len(flux)) * (36.0/(60.0*24)) # time in days
    plt.figure(figsize=(12,4))
    plt.title('Flux of star {} with exoplanets'.format(i+1))
    plt.ylabel('Flux')
    plt.xlabel('Time in days')
    plt.plot(time, flux)


# In[11]:

#Analyzing a star with exoplanet

i = 2
flux1 = exotrain[exotrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
time = np.arange(len(flux1)) * (36.0/(60.0*24.0)) # time in days
plt.figure(figsize=(12,4))
plt.ylabel('Flux')
plt.xlabel('Time in days')
plt.plot(time, flux1)


# In[12]:

#Applying Gausian filter to the flux data

i = 2
flux2 = ndimage.filters.gaussian_filter(flux1, sigma=5)
time = np.arange(len(flux2)) * (36.0/(60.0*24)) # time in days
plt.figure(figsize=(12,4))
plt.ylabel('Flux')
plt.xlabel('Time in days')
plt.plot(time, flux2)


# In[13]:

#Detrending the flux

i = 2
flux3 = flux1 - flux2
time = np.arange(len(flux3)) * (36.0/(60.0*24)) # time in days
plt.figure(figsize=(15,5))
plt.ylabel('Flux')
plt.xlabel('Time in days')
plt.plot(time, flux3)


# In[14]:

#Normalizing the flux with exoplanet

i = 2
flux3normalized = (flux3-np.mean(flux3))/(np.max(flux3)-np.min(flux3))
time = np.arange(len(flux3normalized)) * (36.0/(60.0*24)) # time in days
plt.figure(figsize=(12,4))
plt.ylabel('Normalized flux')
plt.xlabel('Time in days')
plt.plot(time, flux3normalized)


# In[15]:

#function for detrending and normalizing

def detrender_normalizer(X):
    flux1 = X
    flux2 = ndimage.filters.gaussian_filter(flux1, sigma=5)
    flux3 = flux1 - flux2
    flux3normalized = (flux3-np.mean(flux3)) / (np.max(flux3)-np.min(flux3))
    return flux3normalized


# In[16]:

#detrending the whole data

exotrain.iloc[:,1:] = exotrain.iloc[:,1:].apply(detrender_normalizer,axis=1)
exotest.iloc[:,1:] = exotest.iloc[:,1:].apply(detrender_normalizer,axis=1)


# In[17]:

#exotrain.head()
exotest.head()


# In[18]:

#clipping the data using sigma_clip of astropy package

from astropy.io import fits
from astropy import stats
from astropy.stats import sigma_clip

filtered_data = sigma_clip(data_norm, sigma=3, iters=10)


# In[19]:

fluxes_filtered=filtered_data[0,:]
time = np.arange(len(flux3normalized)) * (36.0/(60.0*24)) # time in days
plt.figure(figsize=(12,4))
plt.plot(time,fluxes_filtered)


# In[20]:

#function for removing the outliers

def reduce_upper_outliers(df,reduce = 0.01, half_width=4):
   
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values:
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
        for j in range(remove):
            idx = sorted_values.index[j]
            #print(idx)
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            #print(idx,idx_num)
            for k in range(2*half_width+1):
                idx2 = idx_num + k - half_width
                if idx2 <1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.'+str(idx2)] # corrected from 'FLUX-' to 'FLUX.'
                
                count += 1
            new_val /= count # count will always be positive here
            #print(new_val)
            if new_val < values[idx]: # just in case there's a few persistently high adjacent values
                df.set_value(i,idx,new_val)
        
    return df       


# In[21]:

#applying the function to whole data

exotrain.iloc[:,1:] = reduce_upper_outliers(exotrain.iloc[:,1:])
exotest.iloc[:,1:] = reduce_upper_outliers(exotest.iloc[:,1:])


# In[22]:

#plotting the fluxes after removing upper outliers

i = 9
flux1 = exotrain[exotrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
flux1 = flux1.reset_index(drop=True)
time = np.arange(len(flux1)) * (36.0/(60.0*24)) # time in units of hours
plt.figure(figsize=(15,5))
plt.ylabel('flux')
plt.xlabel('Time in days')
plt.plot(time, flux1)


# In[23]:

#Exploring the bls data

fluxes=data.iloc[0,:]
qmi = 0.01
qma = 0.1
fmin = 0.3 
df = 0.001 
nf = 1000
nb = 200
res = bls(t, fluxes, qmi, qma, fmin, df, nf, nb)
print "Best SR: ", res[0], "\nIngress: ", res[1], "\nEgress: ", res[2], "\nq: ", res[3], "\nDepth: ", res[4], "\nPeriod: ", res[5], "\nSDE: ", res[6]


# In[24]:

fluxes1=data.iloc[100,:]

qmi = 0.01
qma = 0.1
fmin = 0.3 
df = 0.001 
nf = 1000
nb = 200
res1 = bls(t, fluxes1, qmi, qma, fmin, df, nf, nb)
print "Best SR: ", res1[0], "\nIngress: ", res1[1], "\nEgress: ", res1[2], "\nq: ", res1[3], "\nDepth: ", res1[4], "\nPeriod: ", res1[5], "\nSDE: ", res1[6]


# In[25]:

#v- variability of the fluxes from mean 
#with exoplanet

t1 = t[0]
u = t - t1
s = np.mean(fluxes)
v = fluxes - s
plt.figure(figsize=(15,5))
plt.plot(u, v, 'b.')
plt.title("Data centering of flux with exoplanet")
plt.xlabel(r"$t - t_0$")
plt.ylabel(r"$x(t) - \mu$")


# In[26]:

#without exoplanet

t1 = t[0]
u = t - t1
s = np.mean(fluxes1)
v = fluxes1 - s
plt.figure(figsize=(15,5))
plt.plot(u, v, 'b.')
plt.title("Data centering of flux with no exoplanet")
plt.xlabel(r"$t - t_0$")
plt.ylabel(r"$x(t) - \mu$")


# In[27]:

#data folded to the correct period and 200 bins
#with exoplanet

f=fluxes

f0 = 1.0/res[5] #  freq = 1/T
nbin = nb # number of bin
n = len(t)
ibi = np.zeros(nbin)
y = np.zeros(nbin)
phase = np.linspace(0.0, 1.0, nbin)

for i in range(n):
    ph = u[i]*f0 
    ph = ph - int(ph)
    j = int(nbin*ph) # data to a bin 
    ibi[j] = ibi[j] + 1.0 # number of data in a bin
    y[j] = y[j] + v[i] # sum of light in a bin

plt.figure(figsize=(15,5))
plt.plot(phase, y/ibi, 'r.')
plt.title("Period: {0} d  bin: {1}".format(1/f0, nbin))
plt.xlabel(r"Phase ($\phi$)")
plt.ylabel(r"Mean value of $x(\phi)$ in a bin")


# In[28]:

#data folded to the correct period and 200 bins
#with no exoplanet

f=fluxes1

f0 = 1.0/res1[5] #  freq = 1/T
nbin = nb # number of bin
n = len(t)
ibi = np.zeros(nbin)
y = np.zeros(nbin)
phase = np.linspace(0.0, 1.0, nbin)

for i in range(n):
    ph = u[i]*f0 
    ph = ph - int(ph)
    j = int(nbin*ph) # data to a bin 
    ibi[j] = ibi[j] + 1.0 # number of data in a bin
    y[j] = y[j] + v[i] # sum of light in a bin

plt.figure(figsize=(15,5))
plt.plot(phase, y/ibi, 'r.')
plt.title("Period: {0} d  bin: {1}".format(1/f0, nbin))
plt.xlabel(r"Phase ($\phi$)")
plt.ylabel(r"Mean value of $x(\phi)$ in a bin")


# In[29]:

#data only folded not binned

T0 = res[5]
n = len(t)
y = np.zeros(n)
y = t % T0

plt.figure(figsize=(15,5))
plt.plot(y, v, 'r.')
plt.title("Period: {0} d".format(1/f0))
plt.xlabel(r"Phase ($\phi$)")
plt.ylabel(r"Mean value of $x(\phi)$ in a bin")


# In[30]:

#folded with wrong trial period more scattered

f0 = 0.4 #  freq = 1/T
nbin = nb # number of bin
n = len(t)
ibi = np.zeros(nbin)
y = np.zeros(nbin)
phase = np.linspace(0.0, 1.0, nbin)

for i in range(n):
    ph = u[i]*f0 
    ph = ph - int(ph)
    j = int(nbin*ph) # data to a bin 
    ibi[j] = ibi[j] + 1.0 # number of data in a bin
    y[j] = y[j] + v[i] # sum of light in a bin

plt.figure(figsize=(15,5))
plt.plot(phase, y/ibi, 'r.')
plt.title("Period: {0} d  bin: {1}".format(1/f0, nbin))
plt.xlabel(r"Phase ($\phi$)")
plt.ylabel(r"Mean value of $x(\phi)$ in a bin")


# In[31]:

#FFT

import scipy.fftpack as syfp
import scipy as sy
import pylab as pyl
FFT = sy.fft(f)
freqs = syfp.fftfreq(len(u))


# In[32]:

#plot to see what happens at the peak

plt.figure(figsize=(15,8))
pyl.subplot(211)
pyl.plot(t, f)
pyl.xlabel('Time')
pyl.ylabel('Amplitude')
pyl.subplot(212)
pyl.plot(freqs, sy.log10(abs(FFT)), '.')  
pyl.xlim(-.05, .05)                       
pyl.show()


# In[33]:

#since positive labels are too less we use bootstrap and stratified 5 fold in modelling

def model_evaluator(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    
    bootstrapped_accuracies = list()
    bootstrapped_precisions = list()
    bootstrapped_recalls    = list()
    bootstrapped_f1s        = list()
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
                
        df_train    = X_train.join(y_train)
        df_planet   = df_train[df_train.LABEL == 2].reset_index(drop=True)
        df_noplanet = df_train[df_train.LABEL == 1].reset_index(drop=True)
        df_boot     = df_noplanet
                        
        index = np.arange(0, df_planet.shape[0])
        temp_index = np.random.choice(index, size=df_noplanet.shape[0])
        df_boot = df_boot.append(df_planet.iloc[temp_index])
        
        df_boot = df_boot.reset_index(drop=True)
        X_train_boot = df_boot.drop('LABEL', axis=1)
        y_train_boot = df_boot.LABEL
                    
        est_boot = model.fit(X_train_boot, y_train_boot)
        y_test_pred = est_boot.predict(X_test)
        
        bootstrapped_accuracies.append(accuracy_score(y_test, y_test_pred))
        bootstrapped_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))
        bootstrapped_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))
        bootstrapped_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))
    
    print("Average Accuracy", "{:0.10f}".format(np.mean(bootstrapped_accuracies)))
    print("Average Precision:", "{:0.10f}".format(np.mean(bootstrapped_precisions)))
    print("Average Recall:", "{:0.10f}".format(np.mean(bootstrapped_recalls)))
    print("Average F1:", "{:0.10f}".format(np.mean(bootstrapped_f1s)))


# In[34]:

exotrain_raw = pd.read_csv('ExoTrain.csv')


# In[35]:

X_raw = exotrain_raw.drop('LABEL', axis=1)
y_raw = exotrain_raw.LABEL


# In[36]:

X = exotrain.drop('LABEL', axis=1)
y = exotrain.LABEL


# In[37]:

#base model fscore of raw data

model_evaluator(X, y, LinearSVC())


# In[38]:

import scipy


# In[39]:

def fft_data(X):
    Spectrum = scipy.fft(X, n=X.size)
    return np.abs(Spectrum)


# In[40]:

X_train = exotrain.drop('LABEL', axis=1)
y_train = exotrain.LABEL
X_test = exotest.drop('LABEL', axis=1)
y_test = exotest.LABEL


# In[41]:

#fft to whole data

new_X_train = X_train.apply(fft_data,axis=1)
new_X_test = X_test.apply(fft_data,axis=1)


# In[42]:

new_X_test.head()


# In[43]:

y = y_train
X = new_X_train

y_final_test = y_test
X_final_test = new_X_test


# In[44]:

#plotting the data after fft

df = X.join(y)
i = 9
spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
plt.figure(figsize=(15,5))
plt.ylabel('Unitless flux')
plt.xlabel('Frequency, Hz')
plt.plot(freq, spec1)


# In[45]:

#since the plot is symmetrical we take half of the data


X = X.iloc[:,:(X.shape[1]//2)]
X_final_test = X_final_test.iloc[:,:(X_final_test.shape[1]//2)]


# In[46]:

#detailed analysis of plots containing flux data with no exoplanets

df = X.join(y)
for i in [0, 499, 599, 1499]:
    spec1 = df[df.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]
    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
    plt.figure(figsize=(15,5))
    plt.ylabel('Unitless flux')
    plt.xlabel('Frequency, Hz')
    plt.plot(freq, spec1)


# In[47]:

#detailed analysis of plots containing flux data with exoplanets

df = X.join(y)
for i in [0,4,9]:
    spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
    plt.figure(figsize=(15,5))
    plt.ylabel('Unitless flux')
    plt.xlabel('Frequency, Hz')
    plt.plot(freq, spec1)


# In[48]:

from sklearn.preprocessing import normalize


# In[49]:

#normalizing the data

X = pd.DataFrame(normalize(X))
X_final_test = pd.DataFrame(normalize(X_final_test))


# In[50]:

# after normalizing analyzing the plots with exoplanets

df = X.join(y)
for i in [0,4,9]:
    spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
    plt.figure(figsize=(15,5))
    plt.ylabel('Unitless flux')
    plt.xlabel('Frequency, Hz')
    plt.plot(freq, spec1)


# In[51]:

#now using the features extracted using bls algorithm

features= pd.read_csv("data.csv")
features_test= pd.read_csv("Features_test.csv")


# In[52]:

features_test.shape


# In[53]:

features_norm = pd.DataFrame(normalize(features))
features_norm_test = pd.DataFrame(normalize(features_test))


# In[54]:

#using incremental pca

from sklearn.decomposition import IncrementalPCA


# In[55]:

ipca = IncrementalPCA(n_components=165)


# In[56]:

ipca.fit(X)
X_pca=ipca.transform(X)

ipca.fit(X_final_test)
X_pca_test=ipca.transform(X_final_test)


# In[57]:

X_pca=pd.DataFrame(X_pca)
X_pca_test=pd.DataFrame(X_pca_test)


# In[58]:

X_pca_features=pd.concat([X_pca,features_norm],axis=1)
X_pca_features_test= pd.concat([X_pca_test,features_norm_test],axis=1)
print(X_pca_features.shape)
print(X_pca_features_test.shape)


# In[60]:

#Linear SVC

model_evaluator(X_pca_features,y,LinearSVC())


# In[61]:

#knn with 5 neighbours

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)

model_evaluator(X_pca_features,y,neigh)


# In[62]:

#gaussian NB

from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
model_evaluator(X_pca_features,y,clf_nb)


# In[63]:

#GBM

from sklearn.ensemble import GradientBoostingClassifier
clf_gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1,max_depth=1, random_state=0)

model_evaluator(X_pca_features,y,clf_gbm)


# In[64]:

#adaboost

clf_ada = AdaBoostClassifier(n_estimators=450, learning_rate=0.1)
model_evaluator(X_pca_features,y,clf_ada)


# In[65]:

#Random forest

clf_rf = RandomForestClassifier(n_estimators=100,max_depth=1,min_samples_split=5)
model_evaluator(X_pca_features,y,clf_rf)


# In[67]:

#bagging

bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=2,weights='distance'),oob_score=False,max_samples=0.5,max_features=1.0)

model_evaluator(X_pca_features,y,bagging)


# In[68]:

y_new = y - 1


# In[69]:

from sklearn.model_selection import RandomizedSearchCV

# develop your "tuned parameters"

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print(__doc__)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y_new, test_size=0.5, stratify=y)

# Set the parameters by cross-validation
tuned_parameters = [{'penalty': ['l2'], #'l1'],
              'loss': ['hinge'],
              'dual': [True],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': ['balanced'],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]},
                    {'penalty': ['l2'], #'l1'],
              'loss': ['squared_hinge'],
              'dual': [True, False],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': ['balanced'],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]},
                   {'penalty': ['l1'],
              'loss': ['squared_hinge'], #'hinge'],
              'dual': [False],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': ['balanced'],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]}]

model_scores = ['precision', 'recall', 'f1']

for model_score in model_scores:
    print("# Tuning hyper-parameters for %s" % model_score)
    print()
    for tuned_parameter in tuned_parameters:
        clf = RandomizedSearchCV(LinearSVC(), tuned_parameter, cv=3, scoring=model_score, n_jobs=-1)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(confusion_matrix(y_true, y_pred))
        print()


# In[72]:

def bootstrap(X,y):
    
        df_train    = X.join(y)
        df_planet   = df_train[df_train.LABEL == 2].reset_index(drop=True)
        df_noplanet = df_train[df_train.LABEL == 1].reset_index(drop=True)
        df_boot     = df_noplanet
                        
        index = np.arange(0, df_planet.shape[0])
        temp_index = np.random.choice(index, size=df_noplanet.shape[0])
        df_boot = df_boot.append(df_planet.iloc[temp_index])
        
        df_boot = df_boot.reset_index(drop=True)
        X_boot = df_boot.drop('LABEL', axis=1)
        y_boot = df_boot.LABEL
        
        return X_boot, y_boot


# In[74]:

X_train, y_train = bootstrap(X, y)


# In[75]:

print(X.shape)
print(X_train.shape)


# In[76]:

print(y.value_counts())
print(y_train.value_counts())


# In[95]:

#best model

model_evaluator(X_pca_features,y, LinearSVC())


# In[96]:

X_boot,y_boot= bootstrap(X_pca_features, y)
print(X_boot.shape)


# In[97]:

final_model = LinearSVC()
final_model.fit(X_boot, y_boot)


# In[98]:

X_pca_features_test.shape


# In[99]:

y_pred = final_model.predict(X_pca_features_test)


# In[101]:

out=pd.DataFrame()
out['LABEL'] = y_pred
out.to_csv('output_final.csv', index=False)


# In[ ]:



