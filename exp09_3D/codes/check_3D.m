function mi_3d=check_3D(features,dist_types)
    nSet=2;
    nL=length(features{1})-1;% Skipping the last layers
    mi_3d=zeros(2,nL);
    %% Analysis
    mi_3d_raw1=zeros(nL,2);mi_3d_raw2=zeros(nL,2);
    for layers=1:nL
        for set=1:nSet
            img_index=(set-1)*6+[1:6];
            f1=vec(cell2mat(features{img_index(1)}(layers)));
            f2=vec(cell2mat(features{img_index(2)}(layers)));
            f3=vec(cell2mat(features{img_index(3)}(layers)));
            f4=vec(cell2mat(features{img_index(4)}(layers)));
            f5=vec(cell2mat(features{img_index(5)}(layers)));
            f6=vec(cell2mat(features{img_index(6)}(layers)));
            % comparing Y and Cuboid
            dI12=distance_calculation(f1,f2,dist_types);
            dI34=distance_calculation(f3,f4,dist_types);
            mi_3d_raw1(layers,set)=(dI34-dI12)./(dI34+dI12);
            
            % Comparing Square+Y and Cuboid
            dI56=distance_calculation(f5,f6,dist_types);
            mi_3d_raw2(layers,set)=(dI56-dI34)./(dI56+dI34);
        end  
    end
     mi_3d(1,:)=nanmean(mi_3d_raw1,2);
     mi_3d(2,:)=nanmean(mi_3d_raw2,2);
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
