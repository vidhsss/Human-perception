function [miLayerwise_selected_mean,miLayerwise_selected_sem]=check_relsize_effect(features_relsize)
nimages=length(features_relsize);
p.nL=length(features_relsize{1})-1;
p.nGroups=12;
p.percentage_selected=0.07; %   percentage of neurons

miLayerwise_selected_mean=zeros(p.nL,1);
miLayerwise_selected_sem=zeros(p.nL,1);

nTetrad=zeros(p.nL,p.nGroups);
array_neuron_stats=zeros(p.nL,3);% neurons per layer, visually active neurons, tetrads showing positive RE

for layerid=1:p.nL
    disp(layerid);
    % getting response and normalizing
    nNeurons=length(vec(cell2mat(features_relsize{1}(layerid))));
    array_neuron_stats(layerid,1)=nNeurons;
    response=zeros(nNeurons,nimages);
    for ind=1:nimages
        response(:,ind)=vec(cell2mat(features_relsize{ind}(layerid)));
    end
    
    % normalizing it
    nresponse=normalize(response','range');
    nresponse=nresponse';
    
    van=find(sum(nresponse,2)>0); %visually active neurons
    nvresponse=nresponse(van,:); % normalized response from visually active neurons
    nvan=length(van);
    array_neuron_stats(layerid,2)=nvan/nNeurons;
    %init
    RE_L=zeros(nvan,p.nGroups);
    selected_tetrads=zeros(nvan,p.nGroups);
    MI_L=zeros(size(nresponse,1),p.nGroups);
    MI_L_average=[];
    count=1;
    MIn=[];
    for group=1:p.nGroups
        count=count+1;
        imag=(group-1)*4+(1:4);
        % Repeating the analysis from paper.
        % selecting visually active neuron. Select the neurons having
        % positive residual error. Checking if those neurons shows the
        
        n_img1=nresponse(van,imag(1));
        n_img2=nresponse(van,imag(2));
        n_img3=nresponse(van,imag(3));
        n_img4=nresponse(van,imag(4));
        temp_sum=sum([n_img1,n_img2,n_img3,n_img4],2);
        n_active_tertads=length(find(temp_sum>0));
        nTetrad(layerid,count)=n_active_tertads;
        % removing the non active tetradsy
        n_img1(index)=NaN;
        n_img2(index)=NaN;
        n_img3(index)=NaN;
        n_img4(index)=NaN;
        mr1=(n_img1+n_img2)/2;mr2=(n_img3+n_img4)/2;
        mc1=(n_img1+n_img3)/2;mc2=(n_img2+n_img4)/2;
        T=[n_img1,n_img2,n_img3,n_img4];
        re=T+mean(T,2)-[mr1, mr1,mr2,mr2]-[mc1, mc2, mc1, mc2];% finding the resudual error
        RE_L(:,count)=sum(abs(re),2);
        %calculating MI per neurons
        d14=abs(n_img1-n_img4);
        d23=abs(n_img2-n_img3);
        MIn(:,count)=(d23-d14)./(d14+d23);
        
        % repeating the default analysis by appending all the neuronal
        % response to form a single feature per image per layer
        img1=response(van,imag(1));
        img2=response(van,imag(2));
        img3=response(van,imag(3));
        img4=response(van,imag(4));
        % modulation index
        d14_avg=norm(img1-img4,2);
        d23_avg=norm(img2-img3,2);
        MI_Layerwise(layerid,count)=(d23_avg-d14_avg)/(d14_avg+d23_avg); % modulation index of a tetrad
        
    end
    % selecting tetrads
    % selecting tetrads
    MIn_selected=MIn(:); %vectroizing the MI of tetrads
    vactorized_RE_L=RE_L(:);
    vactorized_RE_L(isnan(vactorized_RE_L))=-9999999;  % representing NaN as a large negative value, This is done to match the indexing
    [tempV,tempI]=sort(vactorized_RE_L,'descend'); % sorting the vectroized RI
    count_active_tetrads=sum( nTetrad(layerid,:));
    
    %------------ Selection between fraction of tetrad and all tetrads
    % showing positive interaction.
    temp_number=floor(p.percentage_selected*count_active_tetrads);
    %     temp_number=length(find(tempV>0)); % selecting all tetrad having positive residual error
    array_neuron_stats(layerid,3)=temp_number/(nNeurons*p.nGroups);
    %----------------
    selected_tetrads=selected_tetrads(:);
    selected_tetrads(tempI(1:temp_number))=1;%tempV(1:p.SEL_TETRAD);% weighting the selected tetrad with their RE
    MIn_selected(selected_tetrads(:)==0)=[];
    
    %%%%
    fprintf('\nAmong Selected ( max RE = %.2f min RE = %.2f ) \n',max(tempV(1:temp_number)),min(tempV(1:temp_number)))
    %%%%%%
    % mean and SEM
    miLayerwise_selected_mean(layerid)=nanmean(MIn_selected);
    miLayerwise_selected_sem(layerid)=nansem(MIn_selected);
    %     % plotting the hitogram of modulation index across layers
    %     temp=histogram(MIn_selected,'NumBins',total_bins,'BinLimits',[-1,1],'Normalization','probability');
    %     MI_hist_layer(:,layerid)=temp.Values;
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
