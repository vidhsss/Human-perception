% Checking the results with other distance metrics
% clc;clear all;close all;
% clc;allclear; close all;
% Main Code Directory location and SLASH of the OS
main_folder='/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/';
SLASH='/';
%% Adding Path
addpath([main_folder,'dependencies',SLASH,'matconvnet-1.0-beta24']);
addpath([main_folder,'dependencies',SLASH,'models']);
addpath([main_folder,'dependencies',SLASH,'lib']);
run_path=('/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/dependencies/matconvnet-1.0-beta24/matconvnet-1.0-beta24/matlab/vl_compilenn');
% 
% %% STIM
stim_file_name=sprintf('tatcherFaces.mat');
load(stim_file_name);
% Skip 50 pixels from top row and bottom row t\]\]]]]]]]
% 
% 
% o remove the black region.
S=50;
for i =1:length(stim)
    x=stim{i};
    stim{i}=x(S:end-S,:,:);
end
% %% Networks
type='imagenet-vgg-verydeep-16';
dagg_flag=0;
dist_types={'Euclidean','CityBlock','pearson','spearman'};
time_taken=cell(length(dist_types),1);
MI_across_layers=cell(1,1);
% % MI_across_layers1(1,1)=MI_across_layers(1,1)
% % Effect Reference Level
reference_mi=(4.89-2.92)./(4.89+2.92); % Table-2, Bartlet and Searcy, 1993  
reference_name ='Bartlet and Searcy, 1993';
modelfile=['swav.onnx']
net = importONNXNetwork(modelfile)
% % addpath('/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/dependencies/models');
modelfile2='imagenet-vgg-f.mat'
net1 = load(modelfile2) ;

analyzeNetwork(net)
features=extract_features(stim(1:60),net,1,run_path,net1);
% % % 
% % % % Calcualte the Thatcher Index with different distance metric
N=length(dist_types);
MI_across_layers=cell(N,1);
for iter=1:1
    fprintf('\n Distance Metric = %s \n',dist_types{iter})
    MI_across_layers{iter,1}=CheckThatcherEffect(features,dist_types{iter});
end
% printf(MI_across_layers)
% Plotting the data
mean_data=ones(1,123);
sem_data=ones(1,123);
for ind=1:1
    mean_data(ind,:)=nanmean(MI_across_layers{ind,1},1);
    sem_data(ind,:)=nansem(MI_across_layers{ind,1},1);
end 
% printf(mean_data)
file_name=['..',SLASH,'results',SLASH,'Thatcher Effect Distance Metric Comparison'];
layerwise_mi_figures(mean_data,sem_data,file_name,reference_mi,reference_name,dist_types(1:N),'Thatcher Index');
%% ************** SUBFUNCTIONS *********************************
function thatcherIndex=CheckThatcherEffect(features,dist_type)
nL=length(features{1})-1;% Skipping the last layers
N=15;
features=reshape(features,[N,4]); % Expecting that there are 80 image in total always. 
thatcherIndex=zeros(N,nL);
for ind=1:length(features)
    for L=1:nL
        v1=cell2mat(features{ind,1}(L));
        v2=cell2mat(features{ind,2}(L));
        v3=cell2mat(features{ind,3}(L));
        v4=cell2mat(features{ind,4}(L));
        v12=distance_calculation(v1(:),v2(:),dist_type);
        v34=distance_calculation(v3(:),v4(:),dist_type);
        thatcherIndex(ind,L)=(v12-v34)./(v12+v34);                    %find thatcher index for every layer, for 20 image set, each set having four images 
    end
end
end

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

% $Id: nanmean.m,v 1.1 2004/07/15 22:42:13 glaescher Exp glaescher $