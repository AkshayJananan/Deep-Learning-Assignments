import numpy  as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
A=imread('mccu.jpg')
X_data=np.mean(A,-1)
img=plt.imshow(X_data)
img.set_cmap('gray')
plt.title('Original greyscale image')
plt.show()
####
U,S,VT=np.linalg.svd(X_data,full_matrices=False)
S=np.diag(S)
j=0
for r in (5,10,15,20,50,100):
    img_reduce=U[:,:r] @ S[:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j+=1
    img=plt.imshow(img_reduce)
    img.set_cmap('gray')
    plt.title('r='+str(r))
    plt.show()
    if r==5:
        s1=S[:r,:r]
    if r==10:
        s2=S[:r,:r]
    if r==15:
        s3 = S[:r, :r]
    if r==20:
        s4 = S[:r, :r]
    if r==50:
        s5 = S[:r, :r]
    if r==100:
        s6 = S[:r, :r]
plt.plot(s1,'r.',label='for r=5 values')
plt.plot(s2,'b.',label='for r=10 values')
plt.plot(s3,'g.',label='for r=15 values')
plt.plot(s4,'y.',label='for r=20 values')
plt.plot(s5,'r.',label='for r=50 values')
plt.plot(s6,'b.',label='for r=100 values')
plt.plot(S,label="All sigma values")
plt.show()