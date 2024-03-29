% figure code 3D perception-distance metric comparison
% Credits  : GEORGIN
% clc;allclear; close all;
%% Main Code Directory location and SLASH of the OS

main_folder='/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/';
SLASH='/';
%% Adding Path
addpath([main_folder,'dependencies',SLASH,'matconvnet-1.0-beta24']);
addpath([main_folder,'dependencies',SLASH,'models']);
addpath([main_folder,'dependencies',SLASH,'lib']);
run_path=[main_folder,'dependencies',SLASH,'matconvnet-1.0-beta24',SLASH,'matlab',SLASH,'vl_setupnn'];
%% STIM
file_name_stim='3d.mat';
load(file_name_stim);  % Images
%% Networks
type='imagenet-vgg-verydeep-16';
dagg_flag=0;
dist_types={'Euclidean','CityBlock','pearson','spearman'};
time_taken=cell(length(dist_types),1);
modelfile=['swav.onnx']
net = importONNXNetwork(modelfile)
% addpath('/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/dependencies/models');
modelfile2='imagenet-vgg-f.mat'
net1 = load(modelfile2) ;
%% Effect Reference Level
reference_mi=zeros(2,1);
reference_name=cell(2,1)
reference_mi(1)=0.4;reference_name{1}='Enns & Rensink,1990';
reference_mi(2)=0.76;reference_name{2}='Enns & Rensink,1991';

%% Extract Features
fprintf('\n Extracting Features\n');
features=extract_features(stim,net,1,run_path,net1);

%% Calculate the 3D processing Index
N=length(dist_types);
MI_across_layers=cell(N,1);
for iter=1:N
    fprintf('\n Distance Metric = %s \n',dist_types{iter})
    mi_3d=check_3D(features,dist_types{iter});
    mean_data_condone(iter,:)=mi_3d(1,:);
    mean_data_condtwo(iter,:)=mi_3d(2,:);
end

y_label='3D Index';
Saving_file_name_1=['..',SLASH,'results',SLASH,'Exp09-3D_condition-1,net = VGG16'];
layerwise_mi_figures(mean_data_condone(1,:),[],Saving_file_name_1,reference_mi,reference_name,dist_types,y_label);

Saving_file_name_2=['..',SLASH,'results',SLASH,'Exp09-3D_condition-2,net = VGG16'];
layerwise_mi_figures(mean_data_condtwo(1,:),[],Saving_file_name_2,reference_mi,reference_name,dist_types,y_label);



