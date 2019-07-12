clear ;
close all;

gpus        = 1;

%%
% reset(gpuDevice(gpus));

restoredefaultpath();

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;
%% ncoils = [1, 8]
% if ncoils == 1, fig 7
% else if ncoils == 8, fig 9

ncoils          = 8;    % [1, 8]

%%
dataDir         = './data/';
netDir        	= './network/';

sz              = [320, 256];

dsr             = 3;

if ncoils == 1
    imdb = load('./data/imdb_fig7.mat');
    imdb.images.sensitivity = 1;
    
    ziy_set = [160, 145];
    zix_set = [100, 110];
elseif ncoils == 8
    imdb = load('./data/imdb_fig9.mat');
    
    ziy_set = [90, 150];
    zix_set = [150, 95];
end

%%
networkType     = 'dagnn';

solver_handle	= [];

imageRange      = [0, 100];
imageSize       = [sz(1), sz(2), ncoils];
inputSize       = [sz(1), sz(2), ncoils];

wgt             = 1e-1;
numEpochs       = 500;

batchSize       = 4;
subbatchSize    = 4;
numSubBatches   = ceil(batchSize/subbatchSize);
batchSample     = 1;

lrnrate_f2i  	= logspace(-5, -6, 300);
lrnrate_i2i  	= logspace(-4, -5, 300);
wgtdecay        = 1e-4;

meanNorm        = false;
varNorm         = false;

train           = struct('gpus', gpus);

%%
network_res_unwgt_i2i	= 'cnn_residual_image_domain_learning_init';
expDir_res_unwgt_i2i	= [netDir network_res_unwgt_i2i '_' num2str(ncoils) 'coil'];

network_res_wgt_f2i     = 'cnn_residual_k_space_deep_learning_w_weight_init';
expDir_res_wgt_f2i      = [netDir network_res_wgt_f2i '_' num2str(ncoils) 'coil'];

%%
modelName               = 'net-epoch';
modelPath               = @(epdir, ep) fullfile(epdir, sprintf([modelName '-%d.mat'], ep));

epoch_res_unwgt_i2i   	= findLastCheckpoint(expDir_res_unwgt_i2i, modelName);
[net_res_unwgt_i2i, ~, stats_res_unwgt_i2i] = loadState(modelPath(expDir_res_unwgt_i2i, epoch_res_unwgt_i2i));

epoch_res_wgt_f2i   	= findLastCheckpoint(expDir_res_wgt_f2i, modelName);
[net_res_wgt_f2i, ~, stats_res_wgt_f2i] = loadState(modelPath(expDir_res_wgt_f2i, epoch_res_wgt_f2i));

%%
vid_res_unwgt_i2i	= net_res_unwgt_i2i.getVarIndex('regr_img') ;
net_res_unwgt_i2i.vars(vid_res_unwgt_i2i).precious 	= true ;

vid_res_wgt_f2i     = net_res_wgt_f2i.getVarIndex('regr_img') ;
net_res_wgt_f2i.vars(vid_res_wgt_f2i).precious 	= true ;

%% NETWORK MODE : TEST
mode                    = 'test';         % 'test' / 'normal'

net_res_unwgt_i2i.mode	= mode;
net_res_wgt_f2i.mode	= mode;

%%
opts.wgt        = wgt;
opts.offset     = 0;

opts.imageSize  = imageSize;
opts.inputSize  = inputSize;
opts.kernalSize = [0, 0, 0];

opts.meanNorm	= meanNorm;
opts.varNorm	= varNorm;
opts.batchSize  = 8;
opts.gpus       = gpus;

opts.size       = [sz, ncoils, 2];
opts.set        = 1:2;

opts.nnfft2     = @(x) fftshift(fftshift(fft2(x), 1), 2);
opts.nnifft2  	= @(x) ifft2(fftshift(fftshift(x, 1), 2));

%%
opts.input              = 'input_img';
opts.isweight           = false;
opts.weight             = 1;
opts.vid                = vid_res_unwgt_i2i;
tic;
rec_res_unwgt_i2i_8coil	= recon_cnn4img(net_res_unwgt_i2i, imdb.images.data, opts);
t_img                   = toc;

opts.input              = 'input_img';
opts.isweight           = true;
opts.weight             = weight_mask(sz);
opts.vid                = vid_res_wgt_f2i;
tic;
rec_res_wgt_f2i_8coil	= recon_cnn4img(net_res_wgt_f2i, imdb.images.data, opts);
t_type2                 = toc;

%%
labels                  = sum(imdb.images.labels.*conj(imdb.images.sensitivity), 3);
data                    = sum(imdb.images.data.*conj(imdb.images.sensitivity), 3);
rec_res_unwgt_i2i       = sum(rec_res_unwgt_i2i_8coil.*conj(imdb.images.sensitivity), 3);
rec_res_wgt_f2i         = sum(rec_res_wgt_f2i_8coil.*conj(imdb.images.sensitivity), 3);

%%
wnd_img     = [0, 0.7];
wnd_dif     = [0, 0.7]./5;

zoom_bndy    = 1:80;
zoom_bndx    = 1:64;

%%
for iset = opts.set
    label_               	= abs(labels(:,:,:,iset));
    norval                	= max(label_(:));
    
    label_              	= abs(labels(:,:,:,iset))./norval;
    data_                	= abs(data(:,:,:,iset))./norval;
    rec_res_unwgt_i2i_     	= abs(rec_res_unwgt_i2i(:,:,:,iset))./norval;
    rec_res_wgt_f2i_     	= abs(rec_res_wgt_f2i(:,:,:,iset))./norval;
    
    nmse_data               = nmse(data_, label_);
    nmse_res_unwgt_i2i  	= nmse(rec_res_unwgt_i2i_, label_);
    nmse_res_wgt_f2i        = nmse(rec_res_wgt_f2i_, label_);
    
    psnr_data               = psnr(data_, label_);
    psnr_res_unwgt_i2i  	= psnr(rec_res_unwgt_i2i_, label_);
    psnr_res_wgt_f2i        = psnr(rec_res_wgt_f2i_, label_);
    
    ssim_data               = ssim(data_, label_, 'DynamicRange', wnd_img(2));
    ssim_res_unwgt_i2i  	= ssim(rec_res_unwgt_i2i_, label_, 'DynamicRange', wnd_img(2));
    ssim_res_wgt_f2i        = ssim(rec_res_wgt_f2i_, label_, 'DynamicRange', wnd_img(2));
    
    %% ZOOMED IMAGE

    ziy         = ziy_set(iset) + zoom_bndy;
    zix         = zix_set(iset) + zoom_bndy;
    
    box         = make_box(ziy(1), zix(1), zoom_bndy(end), zoom_bndx(end), size(label_));
    
    %% FULL IMAGE
    figure(1);  colormap gray;
    
    subplot(3,8,[1,10]);    imagesc(abs(label_) + box, wnd_img);                                        axis off ;  title({['Ground Truth (' num2str(ncoils) ' coil)']});
    subplot(3,8,[3,12]);    imagesc(abs(data_) + box, wnd_img);                                         axis off ;  title({['Gaussian (x3)'],           ['NMSE = ' num2str(nmse_data, '%.4e')]});
    subplot(3,8,[5,14]);    imagesc(abs(rec_res_unwgt_i2i_) + box, wnd_img);                            axis off ;  title({['Image-domain learning'],	['NMSE = ' num2str(nmse_res_unwgt_i2i, '%.4e')]});
    subplot(3,8,[7,16]);	imagesc(abs(rec_res_wgt_f2i_) + box, wnd_img);                              axis off ;  title({['Ours (Fig. 2(b))'],        ['NMSE = ' num2str(nmse_res_wgt_f2i, '%.4e')]});
    
    subplot(3,8,17);        imagesc(abs(label_(ziy, zix)), wnd_img);                                    axis off ;  title('Zoomed Image');
    subplot(3,8,19);        imagesc(abs(data_(ziy, zix)), wnd_img);                                     axis off ;  title('Zoomed Image');
    subplot(3,8,21);        imagesc(abs(rec_res_unwgt_i2i_(ziy, zix)), wnd_img);                        axis off ;  title('Zoomed Image');
    subplot(3,8,23);        imagesc(abs(rec_res_wgt_f2i_(ziy, zix)), wnd_img);                          axis off ;  title('Zoomed Image');
    
    subplot(3,8,18);        imagesc(abs(label_(ziy, zix) - label_(ziy, zix)), wnd_dif);                 axis off ;  title('Difference Image');
    subplot(3,8,20);        imagesc(abs(data_(ziy, zix) - label_(ziy, zix)), wnd_dif);                  axis off ;  title('Difference Image');
    subplot(3,8,22);        imagesc(abs(rec_res_unwgt_i2i_(ziy, zix) - label_(ziy, zix)), wnd_dif);     axis off ;  title('Difference Image');
    subplot(3,8,24);        imagesc(abs(rec_res_wgt_f2i_(ziy, zix) - label_(ziy, zix)), wnd_dif);       axis off ;  title('Difference Image');
    
%     drawnow();
    pause();
end