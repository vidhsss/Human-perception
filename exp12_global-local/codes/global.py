import numpy as np
from scipy.spatial.distance import pdist

def check_global_local(features,imagepairDetails):
    nL=len(features[0])-1
    N=len(features)
    g1=imagepairDetails[:,0]
    l1=imagepairDetails[:,1]
    g2=imagepairDetails[:,2]
    l2=imagepairDetails[:,3]
    mean_global_distance=np.zeros(nL)
    mean_local_distance=np.zeros(nL)
    indexG = np.where(l1==l2)
    indexL = np.where(g1==g2)
    for Layer in range(nL):
        temp=np.array(features[0][Layer])
        fl=len(temp.flatten())
        layerF=np.zeros((fl,N))
        for ind_img in range(N):
            layerF[:,ind_img]=np.array(features[ind_img][Layer]).flatten()
        layerwiseDist = pdist(layerF, metric = 'euclidean')
        mean_global_distance[Layer]=np.nanmean(layerwiseDist[indexG])
        mean_local_distance[Layer]=np.nanmean(layerwiseDist[indexL])
    mi_global_local=(mean_global_distance-mean_local_distance)/(mean_global_distance+mean_local_distance)
    return mi_global_local
import scipy.io
import numpy as np
# load('L2_VSmain.mat')
# imagepairDetails=L2_str.Image_Pair_Details;
stim_file_name = '/kaggle/input/perception2/L2_VSmain.mat'
mat = scipy.io.loadmat(stim_file_name)
# stim = mat['stim']
# print('Checking thatcherization')
imagepairDetails=mat['L2_str']['Image_Pair_Details']
stim_file_name = '/kaggle/input/perception2/GL.mat'
mat = scipy.io.loadmat(stim_file_name)
stim=mat['stim']
net=model=model_mae
features=extract_features(stim,net)
mi=check_global_local(features,imagepairDetails[0][0])


fig, ax = plt.subplots()
ax.plot(mi,'o',color='blue')
ax.plot(mi,color='blue')

# plt.ylabel(ylabel)
# plt.xlabel(xlabel)

ax.set_ylim(-1,1)
ax.set_xlim(0,21)


rect3 = matplotlib.patches.Rectangle((0,0 ),
                                  21, 1,
                                  color ='lightgray')
ax.add_patch(rect3)
ax.text(0.8,0.8,'Human Perception',color='black',fontsize=10)

plt.show()