# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:48:41 2015
Demostrates image derivatives
Using Gaussian filter, median filter to remove noise
@author: bxiao
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 22:33:16 2015
# Lecture 4: image dervatives
@author: bxiao
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
import copy as C

l = misc.imread('jump.jpg',flatten=1)
# image derivative
s1 = np.array([1,1])
s =s1[None,:] 
dx = np.array([1,-1])
dy = np.array([1,-1])
dy = dy[None,:]

x = ndimage.convolve1d(l,dx,axis= 0)
gx_I = ndimage.convolve(x,s)

y = ndimage.convolve1d(l,dx,axis= 1)
gy_I = ndimage.convolve(y,s)


# alternatively, you can use np.gradient
gx_I,gy_I = np.gradient(l)[:2]

l = ndimage.gaussian_filter(l, 3)
sx = ndimage.sobel(l, axis=0, mode='constant')
sy = ndimage.sobel(l, axis=1, mode='constant')
sx.shape
sy.shape
sob = np.hypot(sx, sy)

# you can also filter your image with sobel
dx=np.array([[1.0, 0.0, -1.0],[2.0, 0.0, -2.0],[1.0, 0.0, -1.0],])
dy=np.transpose(dx)
fo1=ndimage.convolve(l,dx)
fo2=ndimage.convolve(l,dy)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(gx_I,cmap=plt.cm.gray,interpolation='none')
plt.title('x gradient')
plt.subplot(1,2,2)
plt.imshow(gy_I,cmap=plt.cm.gray,interpolation='none')
plt.title('y gradient')
plt.show()

#plt.figure()
#plt.imshow(sob,cmap=plt.cm.gray,interpolation='none')
#plt.show()
#==============================================================================
# image denosing 
#==============================================================================
l = misc.lena()
l = l[230:310, 210:350]
# adding noise to the images 
noisy = l + 0.4 * l.std() * np.random.random(l.shape)
# adding salt and peper noise to the image 
# adding salt
num_salt = np.ceil(0.05 * l.size * 0.5)
coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in l.shape]
out=C.copy(l)                
out[coords] = 255

# adding pepper
num_pepper = np.ceil(0.05* l.size * (1. - 0.05))
coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in l.shape]
out[coords] = 0
out=out.reshape(l.shape)


# gaussian denoised, 
gauss_denoised = ndimage.gaussian_filter(out, 2)
box_denoised = ndimage.uniform_filter(out, 2)
med_denoised = ndimage.median_filter(out, 3)


plt.figure(figsize=(6,6))

plt.subplot(221)
plt.imshow(out, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('added gaussian noise', fontsize=20)
plt.subplot(222)
plt.imshow(gauss_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('Gaussian filter', fontsize=20)
plt.subplot(223)
plt.imshow(box_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('Box filter', fontsize=20)
plt.subplot(224)
plt.imshow(med_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis('off')
plt.title('Median filter', fontsize=20)

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                    right=1)
plt.show()

# take-home exercise: Implement an median filter yourself

