# % function features=extract_features(stim,net,dagg_flag,run_path,net1)
import torch
import scipy.io
import numpy as np
from numpy import matlib as mb
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import torchvision.models as models
def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook
def extract_features(stim,net):
    nimages=len(stim)
    layers = list(net.features) + list(net.classifier)
    feature=[]
    n, transform = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref01")
    for i in range (nimages):
        bimage_ip=stim[i][0]
        # print(bimage_ip.shape)
        img = cv2.cvtColor(bimage_ip, cv2.COLOR_GRAY2BGR)
        # resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA) 
        data=Image.fromarray(img)
        im=transform(data).unsqueeze(0)
        f=[]
        activations = {}
        layers = list(net.features) + list(net.classifier)

# Define a hook function to get the activations of each layer
        def get_activations(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        for name, layer in zip(net.named_modules(), layers):
            layer.register_forward_hook(get_activations(name))
        net(im)
        # Print the activations of each layer
        for name, activation in activations.items():
            print(activation.shape)
            f.append(activation.numpy()[0])
        feature.append(f)
    return feature
    
#     # for ind in range(3):average_image[:,:,ind]=rgb_values[ind]
#     # average_image=imresize(average_image,Src); 
# # %     nL=length(net.Layers)-1;
#     # Src=net.meta.normalization.imageSize(1:2);% size of the normalized image
#     rgb_values=np.zero(1,3)
#     rgb_values=rgb_values[:]
#     average_image=np.ones(224,224,3);
#     FEATS=[]
#     PREDS=[]
#     for i in range (nimages):
#         bimage_ip=stim[i];
#         if size(bimage_ip,3)==1:
#             bimage_ip = np.matlib.repmat(bimage_ip, 1,1,3)
#         # cimage=imresize(bimage_ip,Src);
#         # cimage=cimage-average_image;

#         # cimage = bsxfun(@minus, cimage, net.meta.normalization.averageImage) ;
#         # cimage=cimage-ones(1,1,3)*128; % abstract images
#         # features{i}=cell(41,1);
# # %         layername=net.Layers(1).Name
# # %         features{i}{1}=activations(net,cimage,layername)
#         print(bimage_ip.shape)
#         features = {}
#         preds = net(bimage_ip)
#         net.global_pool.register_forward_hook(get_features('feats'))
#         FEATS.append(features['feats'].cpu().numpy())
#         PREDS.append(preds.detach().cpu().numpy())
#     return FEATS
# #         %         net.eval({'data', cimage})
# # %         for L=1:nL
# # %             scores(L).x = vec(net.vars(L).value);
# # %         end
# # %         features{i}=scores;
# # %         count=1
# # %         size(features{i}{1})
# # %         X= dlarray(cimage,'SSCB');
        
            
#         # add feats and preds to lists
        
       
# # %         for j=1:124
# # %             layername=net.Layers(j).Name
# # % %             count=count+1;
# # % %              features{i}{j-2}=forward(net, X, 'Outputs',layername);
# # % %             size(features{i}{count})
# # %             features{i}{j}=activations(net, cimage,layername);
# # %         end
# # %         features{i}{count+1}=activations(net,features{i}{3},layername);
# # %         count=count+1;
# # %         features{i}{count+1}=features{i}{count}+ features{i}{count-1};
# # %         count=count+1;
# # %         k=13
# # %         while(k<120)
# # %             s=count;
# # %             for j=k:k+6
# # %                 layername=net.Layers(j).Name
# # %                 count=count+1;
# # %                 features{i}{count}=activations(net,features{i}{count-1},layername);
# # %             end
# # %             count=count+1;
# # %             features{i}{count}=features{i}{s+1}+ features{i}{count-1};
# # %             k=k+7;
# # %         end
# # %         for j=121:123
# # %                 layername=net.Layers(j).Name
# # %                 count=count+1;
# # %                 features{i}{count}=activations(net,features{i}{count-1},layername);
# # %         end


# net, transform = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref01")
# # extract_features(stim,net)
# bimage_ip=stim[0][0]
# fig, ax = plt.subplots()
# im = ax.imshow(stim[0][0], extent=[0, 300, 0, 300])
# # b=np.tile(bimage_ip, (1,1,3))
# # print(b.shape)