%% Part-Summation model with distance metric check using a standard pretrained network
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
%% NETWORK
type='imagenet-vgg-verydeep-16';
dagg_flag=0;
short_name='VGG-16';
%% STIM
file_name_stim='natunat_stim.mat';
load(file_name_stim);
stim=images;
for ind=1:length(stim)
    stim{ind}=padarray(stim{ind},[0,50],0,'both');
end
%% Behavior Effect
reference_mi=  0.16;
reference_name='Pramod and Arun, 2016';
y_label_name='Natural Part Advantage';
dist_types={'Euclidean','CityBlock','pearson','spearman'};
time_taken=cell(length(type),1);
%% Extract Features
modelfile='vgg2.onnx'
net = importONNXNetwork(modelfile)
% addpath('/Users/vipul1/Downloads/CNN-perception-5b36b6c5f5a34ba5927879677f66ea9fd9a39fb7/dependencies/models');
modelfile2='imagenet-vgg-f.mat'
net1 = load(modelfile2) ;

% features=extract_features(stim,net,1,run_path,net1);
%% Calculate the part-summation model with different distance metric
load('L2_natunat.mat');
N=length(dist_types);
MI_across_layers=cell(1,1);
for iter=1:1
    fprintf('\n Distance Metric = %s \n',dist_types{iter})
    [r_natural,r_unnatural]=check_part_summation(features,L2_str,dist_types{iter});
    MI_across_layers{iter,1}=r_natural-r_unnatural;
end
%% Plotting the data
mean_data=[];
sem_data=[];
for ind=1:N
    mean_data(ind,:)=MI_across_layers{ind,1};
end 
file_name=['..',SLASH,'results',SLASH,'Part-Matching_Pramod_Distance Metric Comparison'];
y_lim=[-0.2,0.2];
layerwise_mi_figures_part_summation(mean_data,sem_data,file_name,reference_mi,reference_name,dist_types(1:N),y_label_name,[],y_lim);

function b = padarray(varargin)
%PADARRAY Pad an array.
%   B = PADARRAY(A,PADSIZE) pads array A with PADSIZE(k) number of zeros
%   along the k-th dimension of A.  PADSIZE should be a vector of
%   positive integers.
%
%   B = PADARRAY(A,PADSIZE,PADVAL) pads array A with PADVAL (a scalar)
%   instead of with zeros.
%
%   B = PADARRAY(A,PADSIZE,PADVAL,DIRECTION) pads A in the direction
%   specified by the string DIRECTION.  DIRECTION can be one of the
%   following strings.  
%
%       String values for DIRECTION
%       'pre'         Pads before the first array element along each
%                     dimension .
%       'post'        Pads after the last array element along each
%                     dimension. 
%       'both'        Pads before the first array element and after the
%                     last array element along each dimension.
%
%   By default, DIRECTION is 'both'.
%
%   B = PADARRAY(A,PADSIZE,METHOD,DIRECTION) pads array A using the
%   specified METHOD.  METHOD can be one of these strings:
%
%       String values for METHOD
%       'circular'    Pads with circular repetion of elements.
%       'replicate'   Repeats border elements of A.
%       'symmetric'   Pads array with mirror reflections of itself. 
% 
%   Class Support
%   -------------
%   When padding with a constant value, A can be numeric or logical.
%   When padding using the 'circular', 'replicate', or 'symmetric'
%   methods, A can be of any class.  B is of the same class as A.
%
%   Example
%   -------
%   Add three elements of padding to the beginning of a vector.  The
%   padding elements contain mirror copies of the array.
%
%       b = padarray([1 2 3 4],3,'symmetric','pre')
%
%   Add three elements of padding to the end of the first dimension of
%   the array and two elements of padding to the end of the second
%   dimension.  Use the value of the last array element as the padding
%   value.
%
%       B = padarray([1 2; 3 4],[3 2],'replicate','post')
%
%   Add three elements of padding to each dimension of a
%   three-dimensional array.  Each pad element contains the value 0.
%
%       A = [1 2; 3 4];
%       B = [5 6; 7 8];
%       C = cat(3,A,B)
%       D = padarray(C,[3 3],0,'both')
%
%   See also CIRCSHIFT, IMFILTER.

%   Copyright 1993-2003 The MathWorks, Inc.  
%   $Revision: 1.11.4.3 $  $Date: 2003/08/23 05:53:08 $

[a, method, padSize, padVal, direction] = ParseInputs(varargin{:});

if isempty(a),% treat empty matrix similar for any method

   if strcmp(direction,'both')
      sizeB = size(a) + 2*padSize;
   else
      sizeB = size(a) + padSize;
   end

   b = mkconstarray(class(a), padVal, sizeB);
   
else
  switch method
    case 'constant'
        b = ConstantPad(a, padSize, padVal, direction);
        
    case 'circular'
        b = CircularPad(a, padSize, direction);
  
    case 'symmetric'
        b = SymmetricPad(a, padSize, direction);
        
    case 'replicate'
        b = ReplicatePad(a, padSize, direction);
  end      
end

if (islogical(a))
    b = logical(b);
end

end
%%%
%%% ConstantPad
%%%
function b = ConstantPad(a, padSize, padVal, direction)

numDims = numel(padSize);

% Form index vectors to subsasgn input array into output array.
% Also compute the size of the output array.
idx   = cell(1,numDims);
sizeB = zeros(1,numDims);
for k = 1:numDims
    M = size(a,k);
    switch direction
        case 'pre'
            idx{k}   = (1:M) + padSize(k);
            sizeB(k) = M + padSize(k);
            
        case 'post'
            idx{k}   = 1:M;
            sizeB(k) = M + padSize(k);
            
        case 'both'
            idx{k}   = (1:M) + padSize(k);
            sizeB(k) = M + 2*padSize(k);
    end
end

% Initialize output array with the padding value.  Make sure the
% output array is the same type as the input.
b         = mkconstarray(class(a), padVal, sizeB);
b(idx{:}) = a;
end

%%%
%%% CircularPad
%%%
function b = CircularPad(a, padSize, direction)

numDims = numel(padSize);

% Form index vectors to subsasgn input array into output array.
% Also compute the size of the output array.
idx   = cell(1,numDims);
for k = 1:numDims
  M = size(a,k);
  dimNums = 1:M;
  p = padSize(k);
    
  switch direction
    case 'pre'
       idx{k}   = dimNums(mod(-p:M-1, M) + 1);
    
    case 'post'
      idx{k}   = dimNums(mod(0:M+p-1, M) + 1);
    
    case 'both'
      idx{k}   = dimNums(mod(-p:M+p-1, M) + 1);
  
  end
end
b = a(idx{:});
end
%%%
%%% SymmetricPad
%%%
function b = SymmetricPad(a, padSize, direction)

numDims = numel(padSize);

% Form index vectors to subsasgn input array into output array.
% Also compute the size of the output array.
idx   = cell(1,numDims);
for k = 1:numDims
  M = size(a,k);
  dimNums = [1:M M:-1:1];
  p = padSize(k);
    
  switch direction
    case 'pre'
      idx{k}   = dimNums(mod(-p:M-1, 2*M) + 1);
            
    case 'post'
      idx{k}   = dimNums(mod(0:M+p-1, 2*M) + 1);
            
    case 'both'
      idx{k}   = dimNums(mod(-p:M+p-1, 2*M) + 1);
  end
end
b = a(idx{:});
end
%%%
%%% ReplicatePad
%%%
function b = ReplicatePad(a, padSize, direction)

numDims = numel(padSize);

% Form index vectors to subsasgn input array into output array.
% Also compute the size of the output array.
idx   = cell(1,numDims);
for k = 1:numDims
  M = size(a,k);
  p = padSize(k);
  onesVector = ones(1,p);
    
  switch direction
    case 'pre'
      idx{k}   = [onesVector 1:M];
            
    case 'post'
      idx{k}   = [1:M M*onesVector];
            
    case 'both'
      idx{k}   = [onesVector 1:M M*onesVector];
  end
end
 b = a(idx{:});
end
%%%
%%% ParseInputs
%%%
function [a, method, padSize, padVal, direction] = ParseInputs(varargin)

% default values
a         = [];
method    = 'constant';
padSize   = [];
padVal    = 0;
direction = 'both';

% checknargin(2,4,nargin,mfilename);

a = varargin{1};

padSize = varargin{2};
% checkinput(padSize, {'double'}, {'real' 'vector' 'nonnan' 'nonnegative' ...
%                    'integer'}, mfilename, 'PADSIZE', 2);

% Preprocess the padding size
if (numel(padSize) < ndims(a))
    padSize           = padSize(:);
    padSize(ndims(a)) = 0;
end

if nargin > 2

    firstStringToProcess = 3;
    
    if ~ischar(varargin{3})
        % Third input must be pad value.
        padVal = varargin{3};
%        checkinput(padVal, {'numeric' 'logical'}, {'scalar'}, ...
%                   mfilename, 'PADVAL', 3);
        
        firstStringToProcess = 4;
        
    end
    
    for k = firstStringToProcess:nargin
        validStrings = {'circular' 'replicate' 'symmetric' 'pre' ...
                        'post' 'both'};
        string = checkstrs(varargin{k}, validStrings, mfilename, ...
                           'METHOD or DIRECTION', k);
        switch string
         case {'circular' 'replicate' 'symmetric'}
          method = string;
          
         case {'pre' 'post' 'both'}
          direction = string;
          
         otherwise
          error('Images:padarray:unexpectedError', '%s', ...
                'Unexpected logic error.')
        end
    end
end
    
% Check the input array type
if strcmp(method,'constant') && ~(isnumeric(a) || islogical(a))
    id = sprintf('Images:%s:badTypeForConstantPadding', mfilename);
    msg1 = sprintf('Function %s expected A (argument 1)',mfilename);
    msg2 = 'to be numeric or logical for constant padding.';
    error(id,'%s\n%s',msg1,msg2);
end
end

function out = checkstrs(in, valid_strings, function_name, ...
                         variable_name, argument_position)
%CHECKSTRS Check validity of option string.
%   OUT = CHECKSTRS(IN,VALID_STRINGS,FUNCTION_NAME,VARIABLE_NAME, ...
%   ARGUMENT_POSITION) checks the validity of the option string IN.  It
%   returns the matching string in VALID_STRINGS in OUT.  CHECKSTRS looks
%   for a case-insensitive nonambiguous match between IN and the strings
%   in VALID_STRINGS.
%
%   VALID_STRINGS is a cell array containing strings.
%
%   FUNCTION_NAME is a string containing the function name to be used in the
%   formatted error message.
%
%   VARIABLE_NAME is a string containing the documented variable name to be
%   used in the formatted error message.
%
%   ARGUMENT_POSITION is a positive integer indicating which input argument
%   is being checked; it is also used in the formatted error message.

%   Copyright 1993-2003 The MathWorks, Inc.
%   $Revision: 1.3.4.4 $  $Date: 2003/05/03 17:51:45 $

% Except for IN, input arguments are not checked for validity.

% checkinput(in, 'char', 'row', function_name, variable_name, argument_position);

start = regexpi(valid_strings, ['^' in]);
if ~iscell(start)
    start = {start};
end
matches = ~cellfun('isempty',start);
idx = find(matches);

num_matches = length(idx);

if num_matches == 1
    out = valid_strings{idx};

else
    out = substringMatch(valid_strings(idx));
    
    if isempty(out)
        % Convert valid_strings to a single string containing a space-separated list
        % of valid strings.
        list = '';
        for k = 1:length(valid_strings)
            list = [list ', ' valid_strings{k}];
        end
        list(1:2) = [];
        
        msg1 = sprintf('Function %s expected its %s input argument, %s,', ...
                       upper(function_name), num2ordinal(argument_position), ...
                       variable_name);
        msg2 = 'to match one of these strings:';
        
        if num_matches == 0
            msg3 = sprintf('The input, ''%s'', did not match any of the valid strings.', in);
            id = sprintf('Images:%s:unrecognizedStringChoice', function_name);
            
        else
            msg3 = sprintf('The input, ''%s'', matched more than one valid string.', in);
            id = sprintf('Images:%s:ambiguousStringChoice', function_name);
        end
        
        error(id,'%s\n%s\n\n  %s\n\n%s', msg1, msg2, list, msg3);
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function str = substringMatch(strings)
%   STR = substringMatch(STRINGS) looks at STRINGS (a cell array of
%   strings) to see whether the shortest string is a proper substring of
%   all the other strings.  If it is, then substringMatch returns the
%   shortest string; otherwise, it returns the empty string.

if isempty(strings)
    str = '';
else
    len = cellfun('prodofsize',strings);
    [tmp,sortIdx] = sort(len);
    strings = strings(sortIdx);
    
    start = regexpi(strings(2:end), ['^' strings{1}]);
    if isempty(start) || (iscell(start) && any(cellfun('isempty',start)))
        str = '';
    else
        str = strings{1};
    end
end
end


 function out = mkconstarray(class, value, size)
%MKCONSTARRAY creates a constant array of a specified numeric class.
%   A = MKCONSTARRAY(CLASS, VALUE, SIZE) creates a constant array 
%   of value VALUE and of size SIZE.

%   Copyright 1993-2003 The MathWorks, Inc.  
%   $Revision: 1.8.4.1 $  $Date: 2003/01/26 06:00:35 $

out = repmat(feval(class, value), size);
 end