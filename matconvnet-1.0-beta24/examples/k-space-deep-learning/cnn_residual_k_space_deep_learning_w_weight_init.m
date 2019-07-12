function net = cnn_residual_wgt_f2i_unet_init(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NETWORK	: CNP
% PAPER     : Convolutional Neural Pyramid for Image Processing
%               (https://arxiv.org/pdf/1704.02071.pdf)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.meanNorm           = true ;
opts.varNorm            = true ;
opts.networkType        = 'dagnn' ;

opts.method             = 'image';

opts.param              = [];

[opts, ~]               = vl_argparse(opts, varargin) ;

opts.imageRange         = [0, 1];
opts.imageSize          = [256, 256, 1];

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
% opts.nnfft2             = @(x) fftshift(fftshift(fft2(ifftshift(ifftshift(x, 1), 2)), 1), 2);
% opts.nnifft2            = @(x) ifftshift(ifftshift(ifft2(fftshift(fftshift(x, 1), 2)), 1), 2);

[opts, ~]               = vl_argparse(opts, varargin) ;

%%
ch      = opts.inputSize(3);

% flts    = [3, 3, ch, 256];      hflts   = floor((flts(1) - 1)/2)*ones(4, 1);
% flt0	= [3, 3, 256, 256];
% flt1	= [3, 3, 256, 256];
% flt2	= [3, 3, 256, 256];
% flt3	= [3, 3, 256, 256];
% flte    = [1, 1, 256, ch];

% flt0    = [3, 3, ch, 64];
% flt1	= [3, 3, 64, 128];
% flt2	= [3, 3, 128, 256];
% flt3	= [3, 3, 256, 512];
% flte    = [1, 1, 64, ch];

flt0    = [3, 3, 2*ch, 64];
flt1	= [3, 3, 64, 128];
flt2	= [3, 3, 128, 256];
flt3	= [3, 3, 256, 512];
flte    = [1, 1, 64, 2*ch];

fltc    = [1, 1, 2*2*ch, 2*ch];


%%
opts.bBias      = true;
opts.bBnorm     = true;
opts.bReLU      = true;
opts.nConnect   = 2;        % +1 : sum, +2 : concat
opts.numStage   = 4;
opts.scope      = [];

net             = dagnn.DagNN();

%% FFT PATH
l_fft               = dagnn.FFT('nnfft2', opts.param.nnfft2, 'nnifft2', opts.param.nnifft2);
net.addLayer(['l_fft' ], 	l_fft, {'input_img'}, {'input_fft'});

l_wgt            	= dagnn.Weighting('wgt', opts.param.wgt);
net.addLayer('l_wgt', 	l_wgt, {'input_fft', 'dc_fft'}, {'input_wfft'});

%% FFT DOMAIN
opts.input          = 'input_wfft';

%% [STAGE 0] CONTRACT & EXTRACT PATH
nstg                = 0;
[net, lastLayer]	= add_block_multi_img(net, nstg, flt0, opts);

%% [STAGE 1] CONTRACT & EXTRACT PATH
nstg                = 1;
net                 = add_block_multi_img(net, nstg, flt1, opts);

%% [STAGE 2] CONTRACT & EXTRACT PATH
nstg                = 2;
net                 = add_block_multi_img(net, nstg, flt2, opts);

%% [STAGE 3] CONTRACT & EXTRACT PATH
nstg                = 3;
net                 = add_block_multi_img(net, nstg, flt3, opts);

%% FULLY CONNECTED PATH
hcfy        = floor((flte(1) - 1)/2);
hcfx        = floor((flte(2) - 1)/2);

hcf         = [hcfy, hcfy, hcfx, hcfx]; % [TOP BOTTOM LEFT RIGHT]

l_fc_conv          	= dagnn.Conv('size', flte, 'pad', hcf, 'stride', 1, 'hasBias', true);
net.addLayer('l_fc_conv', 	l_fc_conv, {lastLayer}, {'reg_wfft'}, {'fc_cf', 'fc_cb'});

l_sum               = dagnn.Sum();
net.addLayer('l_sum', l_sum, {'input_wfft', 'reg_wfft'}, {'regr_wfft'});

l_uwgt            	= dagnn.UnWeighting('wgt', opts.param.wgt);
net.addLayer('l_uwgt', 	l_uwgt, {'regr_wfft', 'dc_fft'}, {'regr_fft'});

l_ifft              = dagnn.IFFT('nnfft2', opts.param.nnfft2, 'nnifft2', opts.param.nnifft2);
net.addLayer(['l_ifft' ], 	l_ifft, {'regr_fft'}, {'regr_img'});

%%
l_loss_img          = dagnn.EuclideanLoss('p', 2);

net.addLayer('loss_img',    l_loss_img,	{'regr_img', 'label_img'},      {'objective'});

%% YOU HAVE TO RUN THE FUNCTION for INITIALIZATION OF PARAMETERS
net.initParams();

%% Meta parameters
net.meta.inputSize                  = opts.inputSize ;

net.meta.trainOpts.method           = opts.method;

if length(opts.lrnrate) == 2
    net.meta.trainOpts.learningRate     = logspace(opts.lrnrate(1), opts.lrnrate(2), opts.numEpochs) ;
else
    net.meta.trainOpts.learningRate     = opts.lrnrate;
end

net.meta.trainOpts.errorFunction	= 'euclidean';

net.meta.trainOpts.numEpochs        = opts.numEpochs ;
net.meta.trainOpts.batchSize        = opts.batchSize ;
net.meta.trainOpts.numSubBatches    = opts.numSubBatches ;
net.meta.trainOpts.batchSample      = opts.batchSample ;

net.meta.trainOpts.weightDecay      = opts.wgtdecay;
net.meta.trainOpts.momentum         = 9e-1;

net.meta.trainOpts.imageRange    	= opts.imageRange;

net.meta.trainOpts.solver           = opts.solver ;
net.meta.trainOpts.param            = opts.param ;

%%
for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.WaveDec') || isa(net.layers(l).block, 'dagnn.WaveRec')
    k = net.getParamIndex(net.layers(l).params{1}) ;
    net.params(k).learningRate = 0 ;
  end
end

