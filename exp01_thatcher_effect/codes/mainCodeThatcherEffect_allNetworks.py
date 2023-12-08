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
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
    # n, transform = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref01")
    for i in range (nimages):
        bimage_ip=stim[i][0]
        # print(bimage_ip.shape)
        img = cv2.cvtColor(bimage_ip, cv2.COLOR_GRAY2BGR)
        # resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA) 
        # data=Image.fromarray(img)
        # im=transform(data).unsqueeze(0)
        im=transform(img)
        im=im.unsqueeze(0)
        f=[]
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
def CheckThatcherEffect(features):

  nL=len(features[0])
  # % print(features{1})
  N=20
  # features=features.reshape(N,4) 
  dist_type='Euclidean'
  thatcherIndex=np.zeros([N,nL])
  for ind in range (20):
      for L in range(nL):
        v1=features[ind][L][0].flatten();
        v2=features[ind+20][L][0].flatten();
        v3=features[ind+40][L][0].flatten();
        v4=features[ind+60][L][0].flatten();
        v12=np.linalg.norm(v1-v2,2)
        v34=np.linalg.norm(v3-v4,2)    
        thatcherIndex[ind][L]=(v12-v34)/(v12+v34)                 
    # return thatcherIndex            
  return thatcherIndex
def distance_calculation(f1,f2,type):
# % [irow icol d]=size(f1);
# % temp=reshape(permute(f1,[2,1,3]),[irow*icol,d]);
# % [irow icol d]=size(f2);
# % temp2=reshape(permute(f2,[2,1,3]),[irow*icol,d]);
# % X=[temp;temp2];
  f1=f1.T
  f2=f2.T
  X=np.concatenate((f1.T, f2.T))
  if type =='Euclidean':
          # % d=norm(X(1,:)-X(2,:),2);
      try:
        # d=np.sqrt(np.sum((f1[:] - f2[:]) ** 2))
        d=np.linalg.norm(f1-f2,2)
      except:
        d=np.linalg.norm(X[0,:]-X[1,:])

  elif type =='CityBlock':
      d=np.linalg.norm(X[0,:]-X[1,:],1)
  elif type == 'Cosine':
          # % d=pdist(X,'cosine');
        
      d= cosine(X[0,:],X[1,:])
  # % elif type== 'pearson':
  # %         d=pdist(X,'correlation');
  # % elif type=='spearman' :
  # %          d=pdist(X,'spearman');
  else:
            d=0  
  return d
# main_folder='/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/';
# SLASH='/';
# %% Adding Path
# % addpath(['/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/'])
# % path=main_folder,'dependencies',SLASH,'matconvnet-1.0-beta24']);
# % addpath([main_folder,'dependencies',SLASH,'models']);
# % addpath([main_folder,'dependencies',SLASH,'lib']);
# % run_path=('/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/dependencies/matconvnet-1.0-beta24/matconvnet-1.0-beta24/matlab/vl_compilenn');
# %% STIM
import scipy.io
import numpy as np
stim_file_name = '/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/exp01_thatcher_effect/codes/tatcherFaces.mat'
mat = scipy.io.loadmat(stim_file_name)
stim = mat['stim']
print('Checking thatcherization')
# % S = 50
# % for i in range(len(stim)):
# %     x = stim[i]
# %     stim[i] = x[S:-S, :, :]


# % time_taken=cell(length(type),1);
# sel_colour=colormap('Lines');
# % 
# % % Effect Reference Level
# reference_mi=(4.89-2.92)./(4.89+2.92); % Table-2, Bartlet and Searcy, 1993  
# reference_name ='Bartlet and Searcy, 1993';
# % modelfile='vgg.onnx'
# % net = importONNXNetwork(modelfile)
# % analyzeNetwork(net)
# % % for iter=1:length(type)
# net = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
# net=net.to('cuda')
net = models.vgg16(pretrained=False)
# file="/content/drive/MyDrive/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

# net.features = torch.nn.DataParallel(net.features)
# # net.cuda()
# checkpoint = torch.load(file,map_location=torch.device('cpu'))
# net.load_state_dict(checkpoint["state_dict"])
features= extract_features(stim,net)
# tstart=tic;
# fprintf('\n Network- %s \n');
# % % Extracting Features
# fprintf('\n Extracting Features\n');
# % run_path
nL=len(features[0])-1;
print('\n Checking thatcherization\n');
thatcherIndex=CheckThatcherEffect(features);
MI_across_layers=thatcherIndex;
plot(thatcherIndex,'VGG-16','Thatcher Index')

dist_types='Euclidean';
y_label='Thatcher Index';
# Saving_file_name=['..',SLASH,'results',SLASH,'Exp01-TI ',' metric = ',dist_types];
# %     if(iter<=4) % Layer-wise plot for VGG-Network
# layerwise_mi_figures(np.nanmean(MI_across_layers,1),np.nansem(thatcherIndex,1),Saving_file_name,reference_mi,reference_name,dist_types,y_label);
# %     else % layer-wise plot for other network
# %         layerwise_mi_nonVGGfigures(nanmean(thatcherIndex,1),nansem(thatcherIndex,1),Saving_file_name,reference_mi,reference_name,dist_types,y_label);
# %     end
# % end
# % %% PLotting the effect for three networks
# % sel_index=[1,3,4];
# % 
# % N=length(sel_index);
# % mean_data=zeros(N,37);
# % sem_data=zeros(N,37);
# % for ind=1:N  
# %     mean_data(ind,:)=nanmean(MI_across_layers{sel_index(ind),1},1);
# %     sem_data(ind,:)=nansem(MI_across_layers{sel_index(ind),1},1);
# % end
# % file_name=['..',SLASH,'results',SLASH,'Exp01_mainfigure_VGG16_variants'];
# % layerwise_mi_figures(mean_data,sem_data,file_name,reference_mi,reference_name,network_short_name(sel_index),'Thatcher Index');
# % 
# % %% Plotting the effect for starting point during training
# % sel_index=[1,2];
# % N=length(sel_index);
# % mean_data=zeros(N,37);
# % sem_data=zeros(N,37);
# % for ind=1:N  
# %     mean_data(ind,:)=nanmean(MI_across_layers{sel_index(ind),1},1);
# %     sem_data(ind,:)=nansem(MI_across_layers{sel_index(ind),1},1);
# % end
# % file_name=['..',SLASH,'results',SLASH,'Exp01_suppfigure_comparing VGG16 with matconvnet VGG16'];
# % layerwise_mi_figures(mean_data,sem_data,file_name,reference_mi,reference_name,network_short_name(sel_index),'Thatcher Index');

# %%
