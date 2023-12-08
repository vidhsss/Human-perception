% % Mirror Confusion Check from a standard pretrained network
% % Credits  : GEORGIN
% % clc;allclear; close all;
% % Main Code Directory location and SLASH of the OS
% main_folder='/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/';
stim_file_name = '/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/exp01_thatcher_effect/codes/tatcherFaces.mat'

%% Adding Path
addpath([main_folder,'dependencies',SLASH,'matconvnet-1.0-beta24']);
addpath([main_folder,'dependencies',SLASH,'models']);
addpath([main_folder,'dependencies',SLASH,'lib']);
run_path=('/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/dependencies/matconvnet-1.0-beta24/matconvnet-1.0-beta24/matlab/vl_compilenn');
%% STIM
stim_file_name=sprintf('natural_stim_50_rotated_90.mat');
load(stim_file_name);

%% Networks
type='imagenet-vgg-verydeep-16';
dagg_flag=1;
dist_types={'Euclidean','CityBlock','pearson','spearman'};

network_short_name='VGG-16';
%% MAIN CODE
time_taken=cell(length(type),1);
MI_across_layers=cell(length(type),1);
%% Reference
reference_mi=(16.075-14.33)/(16.075+14.33);
reference_name='Rollenhagen & Olson, 2000';

% Extracting Features and calculating mirror confusion index
tic
fprintf('\n Network- %s \n',type);
mirror_confusion_index=CheckMirrorConfusion(stim,type,dagg_flag,run_path,dist_types);
% toc

%% Plotting the effect for three networks
sel_index=1:4;
N=length(sel_index);mean_data=zeros(N,40);
% mean_data=zeros(N,122);
sem_data=zeros(N,40);
for ind=1:1
    mean_data(ind,:)=nanmean(mirror_confusion_index(:,:,sel_index(ind)),1);
    sem_data(ind,:)=nansem(mirror_confusion_index(:,:,sel_index(ind)),1);
end
file_name=sprintf('..%sresults%sExp02_VGG16 distance_metric_comparison',SLASH,SLASH);
layerwise_mi_figures(mean_data,sem_data,[],reference_mi,reference_name,dist_types(sel_index),'Mirror Confusion Index');

%% Sub Functions
function [features,net,net1]=extract_features(net,stim,type,dagg_flag,run_path,net1)
if(dagg_flag==0)
    if(isempty(net))
        run(run_path);
        modelfile2='imagenet-vgg-f.mat'
        net= load(modelfile2)
    end
    nimages=length(stim);
    features=cell(nimages,1);
    Src=net.meta.normalization.imageSize(1:2);% size of the normalized image
    rgb_values=net.meta.normalization.averageImage;rgb_values=rgb_values(:);
    average_image=ones(Src(1),Src(2),3);
    for ind=1:3,average_image(:,:,ind)=rgb_values(ind);end
    for ind=1:nimages
        bimage_ip=single(stim{ind});
        if size(bimage_ip,3)==1, bimage_ip = repmat(bimage_ip,1,1,3); end
        cimage=imresize(bimage_ip,Src);
        cimage=cimage-average_image;
        features{ind}=vl_simplenn(net,cimage);
    end
elseif(dagg_flag==1)
    if(isempty(net))
%         run(run_path);
        modelfile='ipcl.onnx'
        net = importONNXNetwork(modelfile)
        modelfile2='imagenet-vgg-f.mat'
        net1 = load(modelfile2)
    end
    nimages=length(stim);
    features=cell(nimages,1);
    Src=net1.meta.normalization.imageSize(1:2);% size of the normalized image
    rgb_values=net1.meta.normalization.averageImage;rgb_values=rgb_values(:);
    average_image=ones(Src(1),Src(2),3);
    for ind=1:3,average_image(:,:,ind)=rgb_values(ind);end
   
    for i=1:nimages
        bimage_ip=single(stim{i});
        if size(bimage_ip,3)==1, bimage_ip = repmat(bimage_ip,1,1,3); end
        cimage=imresize(bimage_ip,Src);
        cimage=cimage-average_image;
%       
%         features{i}=cell(41,1);
        for j=1:41
            layername=net.Layers(j).Name
            features{i}{j}=activations(net,cimage,layername)
        end
%         layername=net.Layers(45).Name
%         features{i}{45}=activations(net,cimage,layername)

    end
end
end

function mirror_confusion_index=CheckMirrorConfusion(stim,type,dagg_flag,run_path,dist_types)
% Here I assume the Stimuli is arrange in a particulr order
% 100 unique stimuli, followed by 100 mirror about y-axis, followed by 100 mirror about x-axis
N=100; % There are 100 unique stimuli
mirror_confusion_index=[]; % Horizonatal , Vertical
net=[];
net1=[];
for img=1:N
    fprintf('%d,',img);
    img_numbers=[img,N+img,2*N+img];
    stim_sub=stim(img_numbers);
    [features,net,net1]=extract_features(net,stim_sub,type,dagg_flag,run_path,net1);
    nL=length(features{1})-1;
    for L=1:nL
        fi=squeeze(cell2mat(features{1}(L)));
        fYm=squeeze(cell2mat(features{2}(L)));
        fXm=squeeze(cell2mat(features{3}(L)));
        for d=1:1  %length(dist_types)
            dYm=distance_calculation(fYm,fi,dist_types{d});% MIRROR ABOUT X-axis
            dXm=distance_calculation(fXm,fi,dist_types{d});% MIRROR ABOUT Y-axis
            mirror_confusion_index(img,L,d)=(dXm-dYm)./(dXm+dYm);
        end
    end
end
end
import numpy as np
def CheckMirrorConfusion(stim,type,dagg_flag,run_path,dist_types):
    N = 100 # specify the value of N
    img_numbers = np.array([])
    stim_sub = np.array([])
    features = np.array([])
    
    net = # specify the value of net
    net1 = # specify the value of net1
    nL = 0
    fi = np.array([])
    fYm = np.array([])
    fXm = np.array([])
    dYm = 0
    dXm = 0
    mirror_confusion_index = np.array([])

    for img in range(N):
        print(img)
        img_numbers = np.array([img, N+img, 2*N+img])
        stim_sub = stim[img_numbers]
        features, net, net1 = extract_features(net, stim_sub)
        nL = len(features[0])-1
        for L in range(1, nL+1):
            fi = np.squeeze(features[0][L-1])
            fYm = np.squeeze(features[1][L-1])
            fXm = np.squeeze(features[2][L-1])
            for d in range(1): 
                dYm = distance_calculation(fYm, fi, dist_types[d]) # MIRROR ABOUT X-axis
                dXm = distance_calculation(fXm, fi, dist_types[d]) # MIRROR ABOUT Y-axis
                mirror_confusion_index[img-1, L-1, d] = (dXm-dYm)/(dXm+dYm)
function [sem,n] = nansem(x,dim)
if(isvector(x)), x = x(:); end
if(~exist('dim')),dim = 1; end

s = std(x,[],dim); 
n = sum(~isnan(x),dim); 
sem = s./sqrt(n); 

end

function y = nanmean(x,dim)
% FORMAT: Y = NANMEAN(X,DIM)
% 
%    Average or mean value ignoring NaNs
%
%    This function enhances the functionality of NANMEAN as distributed in
%    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
%    identical name).  
%
%    NANMEAN(X,DIM) calculates the mean along any dimension of the N-D
%    array X ignoring NaNs.  If DIM is omitted NANMEAN averages along the
%    first non-singleton dimension of X.
%
%    Similar replacements exist for NANSTD, NANMEDIAN, NANMIN, NANMAX, and
%    NANSUM which are all part of the NaN-suite.
%
%    See also MEAN

% -------------------------------------------------------------------------
%    author:      Jan Glscher
%    affiliation: Neuroimage Nord, University of Hamburg, Germany
%    email:       glaescher@uke.uni-hamburg.de
%    
%    $Revision: 1.1 $ $Date: 2004/07/15 22:42:13 $

if isempty(x)
	y = NaN;
	return
end

if nargin < 2
	dim = min(find(size(x)~=1));
	if isempty(dim)
		dim = 1;
	end
end

% Replace NaNs with zeros.
nans = isnan(x);
x(isnan(x)) = 0; 

% denominator
count = size(x,dim) - sum(nans,dim);

% Protect against a  all NaNs in one dimension
i = find(count==0);
count(i) = ones(size(i));

y = sum(x,dim)./count;
y(i) = i + NaN;

end

% $Id: 