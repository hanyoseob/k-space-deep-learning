clear ;
close all;

gpus        = 1;

%%
% reset(gpuDevice(gpus));

restoredefaultpath();

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

%%
dataDir         = './data/';
netDir        	= './network/';

sz              = [320, 256];

ncoils          = 1;
dsr             = 3;


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

lrnrate_f2i  	= logspace(-5, -6, 300);
lrnrate_i2i  	= logspace(-4, -5, 300);
wgtdecay        = 1e-4;

meanNorm        = false;
varNorm         = false;

train           = struct('gpus', gpus);

%%
network_res_unwgt_i2i	= 'cnn_residual_image_domain_learning_init';
expDir_res_unwgt_i2i	= [netDir network_res_unwgt_i2i '_' num2str(ncoils) 'coil'];

network_res_unwgt_f2i	= 'cnn_residual_k_space_deep_learning_wo_weight_init';
expDir_res_unwgt_f2i	= [netDir network_res_unwgt_f2i '_' num2str(ncoils) 'coil'];

network_res_wgt_f2i     = 'cnn_residual_k_space_deep_learning_w_weight_init';
expDir_res_wgt_f2i      = [netDir network_res_wgt_f2i '_' num2str(ncoils) 'coil'];

%%
modelName               = 'net-epoch';
modelPath               = @(epdir, ep) fullfile(epdir, sprintf([modelName '-%d.mat'], ep));

epoch_res_unwgt_i2i   	= findLastCheckpoint(expDir_res_unwgt_i2i, modelName);
[net_res_unwgt_i2i, ~, stats_res_unwgt_i2i] = loadState(modelPath(expDir_res_unwgt_i2i, epoch_res_unwgt_i2i));

epoch_res_unwgt_f2i   	= findLastCheckpoint(expDir_res_unwgt_f2i, modelName);
[net_res_unwgt_f2i, ~, stats_res_unwgt_f2i] = loadState(modelPath(expDir_res_unwgt_f2i, epoch_res_unwgt_f2i));

epoch_res_wgt_f2i   	= findLastCheckpoint(expDir_res_wgt_f2i, modelName);
[net_res_wgt_f2i, ~, stats_res_wgt_f2i] = loadState(modelPath(expDir_res_wgt_f2i, epoch_res_wgt_f2i));

%%
vid_res_unwgt_i2i	= net_res_unwgt_i2i.getVarIndex('regr_img') ;
net_res_unwgt_i2i.vars(vid_res_unwgt_i2i).precious 	= true ;

vid_res_wgt_f2i     = net_res_wgt_f2i.getVarIndex('regr_img') ;
net_res_wgt_f2i.vars(vid_res_wgt_f2i).precious 	= true ;

vid_res_unwgt_f2i	= net_res_unwgt_f2i.getVarIndex('regr_img') ;
net_res_unwgt_f2i.vars(vid_res_unwgt_f2i).precious 	= true ;

%%
net_res_unwgt_i2i.removeLayer('mse_img');
net_res_unwgt_i2i.removeLayer('loss_img');

net_res_wgt_f2i.removeLayer('mse_img');
net_res_wgt_f2i.removeLayer('loss_img');

net_res_unwgt_f2i.removeLayer('mse_img');
net_res_unwgt_f2i.removeLayer('loss_img');

%%
epoch_num	= min([epoch_res_unwgt_i2i, epoch_res_unwgt_f2i, epoch_res_wgt_f2i]);

obj_train_res_unwgt_i2i     = [];
obj_train_res_unwgt_f2i     = [];
obj_train_res_wgt_f2i       = [];
obj_val_res_unwgt_i2i       = [];
obj_val_res_unwgt_f2i       = [];
obj_val_res_wgt_f2i         = [];

mse_train_res_unwgt_i2i     = [];
mse_train_res_unwgt_f2i     = [];
mse_train_res_wgt_f2i       = [];
mse_val_res_unwgt_i2i       = [];
mse_val_res_unwgt_f2i       = [];
mse_val_res_wgt_f2i         = [];

for i = 1:epoch_num
    obj_train_res_unwgt_i2i(i)	= stats_res_unwgt_i2i.train(i).objective;
    obj_train_res_unwgt_f2i(i)	= stats_res_unwgt_f2i.train(i).objective;
    obj_train_res_wgt_f2i(i)	= stats_res_wgt_f2i.train(i).objective;
    
    obj_val_res_unwgt_i2i(i)	= stats_res_unwgt_i2i.val(i).objective;
    obj_val_res_unwgt_f2i(i)	= stats_res_unwgt_f2i.val(i).objective;
    obj_val_res_wgt_f2i(i)      = stats_res_wgt_f2i.val(i).objective;
    
    mse_train_res_unwgt_i2i(i)	= stats_res_unwgt_i2i.train(i).mse_img;
    mse_train_res_unwgt_f2i(i)	= stats_res_unwgt_f2i.train(i).mse_img;
    mse_train_res_wgt_f2i(i)	= stats_res_wgt_f2i.train(i).mse_img;
    
    mse_val_res_unwgt_i2i(i)	= stats_res_unwgt_i2i.val(i).mse_img;
    mse_val_res_unwgt_f2i(i)	= stats_res_unwgt_f2i.val(i).mse_img;
    mse_val_res_wgt_f2i(i)      = stats_res_wgt_f2i.val(i).mse_img;
end

%%
figure(1);
set(gcf,'PaperPositionMode','auto');
plot(obj_train_res_unwgt_i2i,	'b--','LineWidth',  2);	hold on;
plot(obj_train_res_unwgt_f2i,   'g--','LineWidth',  2);
plot(obj_train_res_wgt_f2i,    	'r--','LineWidth',  2);

plot(obj_val_res_unwgt_i2i,     'b-','LineWidth',   2);
plot(obj_val_res_unwgt_f2i,     'g-','LineWidth',   2);
plot(obj_val_res_wgt_f2i,       'r-','LineWidth',   2); hold off;

legend('[Train] Image-domain learning', '[Train] Ours (Fig. 2(a))', '[Train] Ours (Fig. 2(b))', ...
    '[Valid] Image-domain learning', '[Valid] Ours (Fig. 2(a))', '[Valid] Ours (Fig. 2(b))', 'location', 'NorthEast');

title('(a) Cartesian', 'FontSize', 40, 'FontWeight', 'bold');

ylabel('Objective', 'FontSize', 40, 'FontWeight', 'bold');
xlabel('# of epochs', 'FontSize', 40, 'FontWeight', 'bold');
ylim([20, 40]);
grid on;
grid minor;

ax              = gca;
ax.FontSize     = 35;
ax.FontWeight 	= 'bold';
ax.FontName     = 'Adobe';

