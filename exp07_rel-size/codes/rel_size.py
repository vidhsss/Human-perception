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
import cv2
import torchvision.models as models
from skimage.transform import resize
from torchvision import transforms

def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook
def extract_features(stim,net):
    nimages=len(stim)
    # layers = list(net.features) + list(net.classifier)
    feature=[]
    Src = (224,224) # size of the normalized image
    rgb_values = np.zeros((3,1))
    rgb_values = rgb_values.flatten()
    average_image = np.ones((Src[0],Src[1],3))
    print(rgb_values.shape)
    for i in range(3):
        average_image[:,:,i] = rgb_values[i]
#     avg_img= np.zeros((3,1))
#     n, transform = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref01")
#     print(transform)
#     print(average_image.shape)
   


    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
    for i in range (nimages):
        img = stim[i][0]
#         print(img[0:10])
        # print(bimage_ip.shape)
        if(img.shape[2]): 
            a=0
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        plt.imshow(img)
#         print(img[0:10])
#         print(img.shape)
#         img = resize(img, (224, 224 ))
#         print(img.shape)
#         print(img[0:10])
        
#         img = img - average_image
#         print(img[0:10])
#         print(img.shape)
#         img = Image.fromarray((img * 255).astype(np.uint8))
#         img=Image.fromarray(img)
#         print(img[0:10])
#         img.astype(np.double)
#         print(img.dtype)
#         im=torch.from_numpy(img.astype(np.float32))
#         im=im.cuda()
        im=transform(img)
#         im=im.type(torch.DoubleTensor)
        print(im.shape)
        print(im.dtype)
        im=im.unsqueeze(0)
#         im = im.permute(0, 3, 1, 2) # from NHWC to NCHW
#         print(im.shape)
#         print(im.dtype)
#         im=im.type(torch.DoubleTensor)
#         print(im.dtype)
#         net.eval()
        f=[]
#         net=net.foat()
        activations = {}
        # with torch.no_grad():
        #   activations = net(im)
        # for name, module in net.named_modules():
      
        #   # if isinstance(module, torch.nn.modules.conv.Conv2d):
            
        #       activations = module(im)
        #       f.append(activations)
#         layers = list(net.features) + list(net.classifier)

# # Define a hook function to get the activations of each layer
        def get_activations(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        for name, layer in net.named_modules():
            layer.register_forward_hook(get_activations(name))
        print(im.dtype)
        net(im)
#         # Print the activations of each layer
        for name, activation in activations.items():
            # print(activation.shape)
            f.append(activation.cpu().numpy()[0])
        feature.append(f)
    return feature
    
def plot(data,xlabel,ylabel):
  fig, ax = plt.subplots()
  ax.plot(np.nanmean(data,axis=0),'o',color='blue')
  ax.plot(np.nanmean(data,axis=0),color='blue')
  
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)

  ax.set_ylim(-1,1)
  ax.set_xlim(0,37)


  rect3 = matplotlib.patches.Rectangle((0,0 ),
                                      37, 1,
                                      color ='lightgray')
  ax.add_patch(rect3)
  ax.text(0.8,0.8,'Human Perception',color='black',fontsize=10)

  plt.show()

import scipy.io
import numpy as np
stim_file_name = '/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/exp07_rel-size/codes/relSize.mat'
mat = scipy.io.loadmat(stim_file_name)
stim = mat['stim']
# print('Checking thatcherization')
net = models.vgg16(pretrained=True)
features= extract_features(stim,net)
# tstart=tic;
print('\n Network- %s \n');
miLayerwise_selected_mean=check_relsize_effect(features)
# [miLayerwise_selected_mean,miLayerwise_selected_sem]=check_relsize_effect(features_relsize);
print('\n Plotting..\n')

plot(miLayerwise_selected_mean,'VGG-16','Relative size')

def check_relsize_effect(features_relsize):
    nimages = len(features_relsize)
    p = {'nL': len(features_relsize[0]) - 1, 'nGroups': 12, 'percentage_selected': 0.07}  # percentage of neurons

    miLayerwise_selected_mean = np.zeros((p['nL'], 1))
    miLayerwise_selected_sem = np.zeros((p['nL'], 1))

    nTetrad = np.zeros((p['nL'], p['nGroups']))
    array_neuron_stats = np.zeros((p['nL'], 3))  # neurons per layer, visually active neurons, tetrads showing positive RE

    for layerid in range(p['nL']):
        print(layerid)
        # getting response and normalizing
        nNeurons = len( np.ravel(np.array(features_relsize[0][layerid])))
        array_neuron_stats[layerid, 0] = nNeurons
        response = np.zeros((nNeurons, nimages))
        for ind in range(nimages):
            response[:, ind] = np.ravel(np.array(features_relsize[ind][layerid]))

        # normalizing it
        nresponse = (response - np.min(response, axis=0)) / (np.max(response, axis=0) - np.min(response, axis=0))
#         nresponse = nresponse.T
#         print(nresponse.shape)
#         print(np.sum(nresponse, axis=1).shape)
        van = np.where(np.sum(nresponse, axis=1) > 0)# visually active neurons
        nvresponse = nresponse[van] # normalized response from visually active neurons
#         print(len(van))
#         print(van)
#         print(nvresponse)
        nvresponse = nresponse[van[0]]
#         print(nvresponse)
        nvan = len(van[0])
        array_neuron_stats[layerid, 1] = nvan / nNeurons
        # init
        
        # print(len(van))
#         van=van.T
#         print(len(van))
        RE_L = np.zeros((nvan, p['nGroups']))
        selected_tetrads = np.zeros((nvan, p['nGroups']))
        MI_L = np.zeros((nresponse.shape[0], p['nGroups']))
        MI_L_average = []
        count = 0
        MIn = []
        for group in range(1, p['nGroups'] + 1):
            count += 1
            imag = (group - 1) * 4 + np.arange(0, 4)
#             print(imag)
            n_img1=nresponse[van,imag[0]]
            n_img1=n_img1[0].T
#             print(n_img1.shape)
            n_img2=nresponse[van,imag[1]]
            n_img2=n_img2[0].T
#             print(n_img2.shape)
            n_img3=nresponse[van,imag[2]]
            n_img3=n_img3[0].T
            n_img4=nresponse[van,imag[3]]
            n_img4=n_img4[0].T
#             print(np.array([n_img1, n_img2, n_img3, n_img4]).T.shape)
            temp_sum = np.sum(np.array([n_img1, n_img2, n_img3, n_img4]).T, axis=1)
#             print(temp_sum.shape)
            n_active_tertads = len([x for x in temp_sum if x>0])
            nTetrad[layerid, count] = n_active_tertads

            # removing the non active tetrads
            index = np.where(temp_sum <= 0)
#             print(index)
            n_img1[index] = np.nan
            n_img2[index] = np.nan
            n_img3[index] = np.nan
            n_img4[index] = np.nan
#             print(len(n_img1))
            mr1 = (n_img1 + n_img2) / 2
            mr2 = (n_img3 + n_img4) / 2
            mc1 = (n_img1 + n_img3) / 2
            mc2 = (n_img2 + n_img4) / 2
            
            T = np.array([n_img1, n_img2, n_img3, n_img4]).T
            re = np.subtract(T.T , np.mean(T, axis=1).T).T - np.array([mr1, mr1, mr2, mr2]).T - np.array([mc1, mc2, mc1, mc2]).T
            # finding the residual error
            # print(re.shape)
            RE_L[:, count] = np.sum(np.abs(re), axis=1)

            #calculating MI per neurons
            d14 = np.abs(n_img1 - n_img4)
            d23 = np.abs(n_img2 - n_img3)
            MIn.append((d23-d14) / (d14+d23))

            # repeating the default analysis by appending all the neuronal
            # response to form a single feature per image per layer
            img1 = response[van,imag[0]]
            img2 = response[van,imag[1]]
            img3 = response[van,imag[2]]
            img4 = response[van,imag[3]]
            img1=img1[0].T
            img2=img2[0].T
            img3=img3[0].T
            img4=img4[0].T
            # modulation index
            d14_avg = np.linalg.norm(img1 - img4, 2)
            d23_avg = np.linalg.norm(img2 - img3, 2)
            MI_L[layerid, count] = (d23_avg - d14_avg) / (d14_avg + d23_avg)  # modulation index of a tetrad

#         % selecting tetrads
#         % selecting tetrads
        MIn_selected=MIn[:]; 
        vactorized_RE_L=RE_L[:];
        vactorized_RE_L[np.isnan(vactorized_RE_L)] = -9999999
        tempV, tempI = zip(*sorted(enumerate(vactorized_RE_L), key=lambda x: x[1], reverse=True))
        count_active_tetrads = np.sum(nTetrad[layerid,:])

#         %------------ Selection between fraction of tetrad and all tetrads
#         % showing positive interaction.

        temp_number = int(p['percentage_selected']*count_active_tetrads)
        array_neuron_stats[layerid,2] = temp_number / (nNeurons * p['nGroups']) 
        selected_tetrads=selected_tetrads[:];
        selected_tetrads[tempI[0:temp_number]]=1;
        MIn_selected[selected_tetrads[:]==0]=[];

     
        print('\nAmong Selected ( max RE = %.2f min RE = %.2f ) \n',np.max(tempV[0:temp_number]),np.min(tempV[0:temp_number]))
     
#         % mean and SEM
        miLayerwise_selected_mean[layerid]=np.mean(MIn_selected);
#         miLayerwise_selected_sem(layerid)=nansem(MIn_selected);
   
    return miLayerwise_selected_mean