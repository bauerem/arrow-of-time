import skvideo.io
import skvideo.datasets
import numpy as np

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


videodata = skvideo.io.vread("test.mp4")
videodata = videodata.astype(np.float32)
#videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny()) #np.random.random(size=(800, 200, 200, 3)) * 255 #
#print(videodata.shape)
#skvideo.io.vwrite("outputvideo.mp4", videodata)

#videodata = np.random.random(size=(400, 4, 4, 1)) * 255
videodata = videodata.reshape((videodata.shape[0],-1))
t = np.arange(videodata.shape[0])

exog = videodata[:,0]
endog = videodata[:,1:4]
#model = sm.tsa.VARMAX(endog, order=(2,0), trend='n') #, exog=exog)
#res = model.fit(maxiter=100, disp=False)

model = sm.tsa.VAR(endog) #, exog=exog)
res = model.fit(2)

print(res.summary())

plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(t,videodata[:,i])
#plt.plot(t,videodata[:,1])
plt.show()

print(model.coeff)
#print(model.params)

def algorithm_1(f, b, sig1, sig2):  # b=f.T
    """
    f : forward
    b : backward....
    """
