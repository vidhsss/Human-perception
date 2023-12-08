from scipy.spatial.distance import cosine
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib
# % function features=extract_features(stim,net,dagg_flag,run_path,net1)
import torch
import scipy.io
import numpy as np
from numpy import matlib as mb
from matplotlib import pyplot as plt
from PIL import Image
# import cv2
import torchvision.models as models
# from skimage.transform import resize
from torchvision import transforms
def extract_features(stim,net):
    nimages=len(stim[0])
    # layers = list(net.features) + list(net.classifier)
    feature=[]
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
    # n, transform = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref01")
    for i in range (nimages):
        img=stim[0][i]
        img=img.astype(np.uint8)
#         print(img.shape)
#         img = img.resize((224, 224))
#         try:
#             if(img.shape[2]):
#                 a=0
#         except:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# #        
        im=transform(img)
        print(im.shape)
        im=im.unsqueeze(0)
        f=[]
        activations = {}
        im=model.patch_embed(im)
        f.append(im.detach().numpy())
        for b in model.blocks:
            im=b(im)
            f.append(im.detach().numpy())
        im=model.norm(im)
        im=model.decoder_embed(im)
        f.append(im.detach().numpy())
        for i in model.decoder_blocks:
            im=i(im)
            f.append(im.detach().numpy())
        feature.append(f)
    return feature
def check_3D(features, dist_types):
    nSet = 2
    nL = len(features[1])   # Skipping the last layers
    mi_3d = np.zeros((2, nL))

    # Analysis
    mi_3d_raw1 = np.zeros((nL, 2))
    mi_3d_raw2 = np.zeros((nL, 2))
    for layers in range(nL):
        for sete in range(1,nSet+1):
            img_index = (sete - 1) * 6 + np.arange(0,6)
            f1 = np.ravel(np.asarray(features[img_index[0]][layers]))
            f2 = np.ravel(np.asarray(features[img_index[1]][layers]))
            f3 = np.ravel(np.asarray(features[img_index[2]][layers]))
            f4 = np.ravel(np.asarray(features[img_index[3]][layers]))
            f5 = np.ravel(np.asarray(features[img_index[4]][layers]))
            f6 = np.ravel(np.asarray(features[img_index[5]][layers]))
            # comparing Y and Cuboid
            dI12=np.linalg.norm(f1-f2,2)
            dI34=np.linalg.norm(f3-f4,2)
#             dI12 = distance_calculation(f1, f2, dist_types)
#             dI34 = distance_calculation(f3, f4, dist_types)
#             print(dI12,dI34)
            mi_3d_raw1[layers, sete-1] = (dI34 - dI12) / (dI34 + dI12)

            # Comparing Square+Y and Cuboid
#             dI56 = distance_calculation(f5, f6, dist_types)
            dI56=np.linalg.norm(f5-f6,2)
            mi_3d_raw2[layers, sete-1] = (dI56 - dI34) / (dI56 + dI34)

    mi_3d[0, :] = np.nanmean(mi_3d_raw1, axis=1)
    mi_3d[1, :] = np.nanmean(mi_3d_raw2, axis=1)
    return mi_3d
stim_file_name = 'CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/exp09_3D/codes/3d.mat'
mat = scipy.io.loadmat(stim_file_name)
stim = mat['stim']

features= extract_features(stim,net)
mi=check_3D(features, "euclidean")
fig, ax = plt.subplots()
ax.plot(mi[1],'o',color='blue')
ax.plot(mi[1],color='blue')
ax.plot(mi[0],'o',color='red')
ax.plot(mi[0],color='red')
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