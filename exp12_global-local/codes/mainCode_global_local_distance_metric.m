% Checking the results with other distance metrics
% Georgin Jacob,19-7-2020 : FIRST VERSION  
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
file_name_stim='GL.mat';
load(file_name_stim);
%% NETWORK
type={'imagenet-vgg-verydeep-16'};
dagg_flag=0;

%% Behvaioral Effect
reference_mi=0.0975;
reference_name='Jacob and Arun, 2019';
modelfile='swav.onnx'
net = importONNXNetwork(modelfile)
% addpath('/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/dependencies/models');
modelfile2='imagenet-vgg-f.mat'
net1 = load(modelfile2) ;
%% Extract Features
features=extract_features(stim,net,1,run_path,net1);
load('L2_VSmain.mat')
imagepairDetails=L2_str.Image_Pair_Details;
%% Calculate Index for data types
dist_types={'Euclidean'};
nL=length(features{1})-1;
N=length(features);nD=length(dist_types);
g1=imagepairDetails(:,1);l1=imagepairDetails(:,2);
g2=imagepairDetails(:,3);l2=imagepairDetails(:,4);

indexG=find(l1==l2);
indexL=find(g1==g2);
for Layer=1:nL
    fprintf('\nLayer- %d', Layer)
    temp=cell2mat(features{1}(Layer));fl=length(temp(:));
    layerF=zeros(fl,N);
    for ind_img=1:N
        layerF(:,ind_img)=vec(cell2mat(features{ind_img}(Layer)));
    end
    for ind=1:nD
        fprintf('\nDistance Types- %s', dist_types{ind})
        layerwiseDist=distance_calculation_matrix(layerF',dist_types{ind});
        mean_global_distance(ind,Layer)=nanmean(layerwiseDist(indexG));
        mean_local_distance(ind,Layer)=nanmean(layerwiseDist(indexL));
    end
end
mi_global_local=(mean_global_distance-mean_local_distance)./(mean_global_distance+mean_local_distance);
mean_data=mi_global_local;sem_data=[];

file_name_pdf=['..',SLASH,'results',SLASH,'global advantage effect Distance Metric Comparison'];
layerwise_mi_figures(mean_data,sem_data,file_name_pdf,reference_mi,reference_name,dist_types,'Global Advantage Index');
%% ************** SUBFUNCTIONS *********************************
function d=distance_calculation_matrix(X,type)
switch type
    case 'Euclidean'
        d=pdist(X,'euclidean');
    case 'CityBlock'
        d=pdist(X,'cityblock');
    case 'Cosine'
        d=pdist(X,'cosine');
    case 'pearson'
        d=pdist(X,'correlation');
    case 'spearman' 
         d=pdist(X,'spearman');
    otherwise
          d=NAN;  
end
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


function y = pdist (x, distfun, varargin)

%   if (nargin < 1)
%     print_usage ();
%   elseif (nargin > 1) && ...
%         ! (ischar (distfun) || ...
%            strcmp (class(distfun), "function_handle"))
% %     error ("pdist: the distance function must be either a string or a function 
% % handle.");
%   end

  if (nargin < 2)
    distfun = "euclidean";
  end

  if (isempty (x))
    error ("pdist: x cannot be empty");
  elseif (length (size (x)) > 2)
    error ("pdist: x must be 1 or 2 dimensional");
  end

  sx1 = size (x, 1);
  y = []; for i = 1:sx1
    tmpd = feval (distfun, x(i,:), x(i+1:sx1,:), varargin{:});
    y = [y;tmpd(:)];
  end

  end


function d = euclidean(u, v)
  d = sqrt (sum ((repmat (u, size (v,1), 1) - v).^2, 2));
end

% function d = seuclidean(u, v)
%   ## FIXME
%   error("Not implemented")
% endfunction
% 
% function d = mahalanobis(u, v, p)
%   repu = repmat (u, size (v,1), 1);
%   d = (repu - v)' * inv (cov (repu, v)) * (repu - v);
%   d = d.^(0.5);
% end
% 
% function d = cityblock(u, v)
%   d = sum (abs (repmat (u, size(v,1), 1) - v), 2);
% endfunction
% 
% function d = minkowski
%   if (nargin < 3)
%     p = 2;
%   endif
% 
%   d = (sum (abs (repmat (u, size(v,1), 1) - v).^p, 2)).^(1/p);
% endfunction
% 
% function d = cosine(u, v)
%   repu = repmat (u, size (v,1), 1);
%   d = dot (repu, v, 2) ./ (dot(repu, repu).*dot(v, v));
% endfunction
% 
% function d = correlation(u, v)
%   repu = repmat (u, size (v,1), 1);
%   d = cor(repu, v);
% endfunction
% 
% function d = spearman(u, v)
%   repu = repmat (u, size (v,1), 1);
%   d = spearman (repu, v);
% endfunction
% 
% function d = hamming(u, v)
% %   Hamming distance, the percentage of coordinates that differ
%   sv2 = size(v, 2);
%   for i = 1:sv2
%     v(:,i) = (v(:,i) == u(i));
%   end
%   d = sum (v,2)./sv2;
%   end
% 
% function d = jaccard(u, v)
% %   ## Jaccard distance, one minus the percentage of non-zero coordinates
% %   ## that differ
%   sv2 = size(v, 2);
%   for i = 1:sv2
%     v(:,i) = (v(:,i) == u(i)) && (u(i) || v(:,i));
%   endfor
%   d = 1 - sum (v,2)./sv2;
% endfunction
% 
% function d = chebychev(u, v)
%   repu = repmat (u, size (v,1), 1);
%   d = max (abs (repu - v), [], 2);
% endfunction