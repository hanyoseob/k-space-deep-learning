clear ;
close all;

gpus        = 1;

%%
reset(gpuDevice(gpus));

restoredefaultpath();

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

%%
dataDir         = './data/';
netDir        	= './network/';

sz              = [320, 256];

ncoils          = 1;
dsr             = 3;

imdb            = load(['./data/imdb_cartesian_' num2str(ncoils) 'coil.mat']);
load('data/smp_mask.mat');

%%
networkType     = 'dagnn';

solver_handle	= [];

imageRange      = [0, 100];
imageSize       = [sz(1), sz(2), ncoils];
inputSize       = [sz(1), sz(2), ncoils];

wgt             = 1e-1;
numEpochs       = 1000;

batchSize       = 4;
subbatchSize    = 4;
numSubBatches   = ceil(batchSize/subbatchSize);
batchSample     = 1;

lrnrate         = logspace(-5, -6, 300);
wgtdecay        = 1e-4;

meanNorm        = false;
varNorm         = false;

train           = struct('gpus', gpus);

%%
network         = 'cnn_residual_k_space_deep_learning_w_weight_init';

%%
param.isflip    = true;
param.preserve  = true;
param.iswgt     = true;

param.nnfft2    = @(x) fftshift(fftshift(fft2(x), 1), 2);
param.nnifft2  	= @(x) ifft2(fftshift(fftshift(x, 1), 2));

param.size      = inputSize;

if param.iswgt
    param.wgt 	= weight_mask(sz);
else
    param.wgt   = 1;
end

param.support           = single(smp_mask);

expDir                  = [netDir network '_' num2str(ncoils) 'coil_'];

%% TRAIN
[net_train, info_train] = cnn_cartesian( 'param', param, ...
 	'wgt',          wgt,        'meanNorm',     meanNorm,   'varNorm',          varNorm,        ...
    'imdb',         imdb,       'network',      network,	'networkType',      networkType,	...
    'expDir',       expDir,     'solver',       solver_handle,'inputSize',      inputSize,	...
    'imageRange',	imageRange,	'imageSize',    imageSize,	'batchSample',      batchSample,    ...
    'numEpochs',    numEpochs,  'batchSize',    batchSize,	'numSubBatches',    numSubBatches,	...
    'lrnrate',      lrnrate,    'wgtdecay',     wgtdecay,   'train',            train);

return ;
