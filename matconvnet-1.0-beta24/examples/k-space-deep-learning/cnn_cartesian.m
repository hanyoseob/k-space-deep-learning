function [net, info] = cnn_cartesian(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.meanNorm           = true ;
opts.varNorm            = true ;
opts.network            = [] ;
opts.networkType        = 'simplenn' ;
opts.method             = 'image';

opts.param              = [];

[opts, varargin]        = vl_argparse(opts, varargin) ;

% sfx = opts.networkType ;
% if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
% opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline-' sfx]) ;
% [opts, varargin] = vl_argparse(opts, varargin) ;
%
% opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
% opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.expDir             = '';
opts.train              = struct() ;
[opts, varargin]        = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.imdb               = [];

opts.imageRange         = [0, 1];
opts.imageSize          = [512, 512, 1];
opts.inputSize          = [40, 40, 1];
opts.windowSize         = [80, 80, 1];
opts.wrapSize           = [0, 0, 1];

opts.wgt                = 1;

opts.numEpochs          = 1e2;
opts.batchSize          = 16;
opts.numSubBatches      = 1;
opts.batchSample        = 1;

opts.lrnrate            = [-3, -5];
opts.wgtdecay           = 1e-4;

opts.solver             = [];

[opts, ~]               = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

network = str2func(opts.network);
net    	= network( 'param', opts.param, 'batchSample', opts.batchSample, ...
    'wgt',          opts.wgt,           'meanNorm',     opts.meanNorm,      'varNorm',          opts.varNorm,     	...
    'networkType',  opts.networkType,   'method',       opts.method,        'imageRange',       opts.imageRange,	...
    'imaegSize',    opts.imageSize,     'inputSize',	opts.inputSize, 	'wrapSize',         opts.wrapSize,      ...
    'numEpochs',    opts.numEpochs,   	'batchSize',    opts.batchSize,     'numSubBatches',    opts.numSubBatches, ...
    'lrnrate',      opts.lrnrate,       'wgtdecay',     opts.wgtdecay,      'solver',           opts.solver) ;

imdb    = opts.imdb;

% net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
    case 'simplenn',    trainfn = @cnn_train ;
        %     case 'dagnn',       trainfn = @cnn_train_dag ;
    case 'dagnn',       trainfn = @cnn_train_dag_cartesian ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
    case 'dagnn'
        bopts = struct( 'numGpus', numel(opts.train.gpus), 'method', opts.method, ...
            'wgt', opts.wgt, 'meanNorm', opts.meanNorm, 'varNorm', opts.varNorm, ...
            'imageSize', opts.imageSize, 'inputSize', opts.inputSize, 'param', opts.param) ;
        fn = @(x,y,z) getDagNNBatch(bopts,x,y,z) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
% function inputs = getDagNNBatch(opts, imdb, batch)
function inputs = getDagNNBatch(opts, imdb, batch, mode)
% --------------------------------------------------------------------
meanNorm    = opts.meanNorm;
varNorm     = opts.varNorm;
nsz         = opts.imageSize;
patch       = opts.inputSize;

param       = opts.param;

support     = param.support;
wgt         = param.wgt;

isflip      = param.isflip;
iswgt       = param.iswgt;

nnfft2      = param.nnfft2;
nnifft2     = param.nnifft2;

scale       = opts.wgt;
offset      = 0;

nbatch      = length(batch);
ndata       = size(imdb.images.labels, 4);

labels_img  = zeros(patch(1), patch(2), nsz(3), nbatch, 'single');

by      = 1:patch(1);
bx      = 1:patch(2);

for ibatch = 1:nbatch
    
    batch_data              = batch(ibatch);
    batch_labels          	= mod(batch_data - 1, ndata) + 1;

    iy                      = floor(rand(1)*(nsz(1) - patch(1))) + by;
    ix                      = floor(rand(1)*(nsz(2) - patch(2))) + bx;
    
    labels_img_             = imdb.images.labels(iy,ix,:,batch_labels);
    
    if isflip
        if (rand > 0.5)
            labels_img_ = flip(labels_img_, 1);
        end
        
        if (rand > 0.5)
            labels_img_ = flip(labels_img_, 2);
        end
    end
    
    labels_img(:,:,:,ibatch)	= labels_img_;
    
end

images_img      = nnifft2(bsxfun(@times, nnfft2(labels_img), support));

if meanNorm
    means   = mean(mean(mean(images_img, 1), 2), 3);
else
    means   = 0;
end

images_img      = bsxfun(@minus, images_img, means);
labels_img      = bsxfun(@minus, labels_img, means);

if varNorm
    vars   = max(max(max(abs(images_img), [], 1), [], 2), [], 3);
else
    vars    = 1;
end

labels_img      = bsxfun(@times, labels_img, 1./vars);
images_img  	= bsxfun(@times, images_img, 1./vars);

labels_img      = scale.*labels_img + offset;
images_img      = scale.*images_img + offset;

if iswgt
    [idcy, idcx]    = find(wgt(:,:,1) == 0);
    images_fft      = nnfft2(images_img);
    dc_fft          = images_fft(idcy, idcx, :, :);
end

labels_img      = comp2ri(labels_img);
images_img      = comp2ri(images_img);

if opts.numGpus > 0
    labels_img  = gpuArray(labels_img) ;
    images_img	= gpuArray(images_img) ;
end

if iswgt
    inputs	= {'input_img', images_img, 'label_img', labels_img, 'dc_fft', dc_fft} ;
else
    inputs	= {'input_img', images_img, 'label_img', labels_img} ;
end

% % --------------------------------------------------------------------
% function inputs = getDagNNBatch(opts, imdb, batch)
% % --------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% if opts.numGpus > 0
%     images = gpuArray(images) ;
% end
% inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
    'train-labels-idx1-ubyte', ...
    't10k-images-idx3-ubyte', ...
    't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir) ;
end

for i=1:4
    if ~exist(fullfile(opts.dataDir, files{i}), 'file')
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
        fprintf('downloading %s\n', url) ;
        gunzip(url, opts.dataDir) ;
    end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
