%% Step 1. Copy the customized library
copyfile('./matlab', '../../matlab');

%% Step 2. Download the trained network (1 coil)
% Image-domain learning network (1 coil)
network_path    = './network/cnn_residual_image_domain_learning_init_1coil/';
network_name	= [network_path 'net-epoch-1000.mat'];
network_url     = 'https://www.dropbox.com/s/6m4ex3o6l8xftb0/net-epoch-1000.mat?dl=1';

mkdir(network_path);
fprintf('downloading image-domain learning network (1 coil) from %s\n', network_url) ;
websave(network_name, network_url);

% k-space deep learning w/o weighting network (1 coil)
network_path    = './network/cnn_residual_k_space_deep_learning_wo_weight_init_1coil/';
network_name	= [network_path 'net-epoch-1000.mat'];
network_url     = 'https://www.dropbox.com/s/6m4ex3o6l8xftb0/net-epoch-1000.mat?dl=1';

mkdir(network_path);
fprintf('downloading k-space deep learning w/o weighitng network (1 coil) from %s\n', network_url) ;
websave(network_name, network_url);

% k-space deep learning w/ weighting network (1 coil)
network_path    = './network/cnn_residual_k_space_deep_learning_w_weight_init_1coil/';
network_name	= [network_path 'net-epoch-1000.mat'];
network_url     = 'https://www.dropbox.com/s/139bjbzagfyatxu/net-epoch-1000.mat?dl=1';

mkdir(network_path);
fprintf('downloading k-space deep learning w/ weighitng network (1 coil) from %s\n', network_url) ;
websave(network_name, network_url);

%% Step 3. Download the trained network (8 coils)
% Image-domain learning network (8 coil)
network_path    = './network/cnn_residual_image_domain_learning_init_8coil/';
network_name	= [network_path 'net-epoch-500.mat'];
network_url     = 'https://www.dropbox.com/s/07rue8esm2rak81/net-epoch-500.mat?dl=1';

mkdir(network_path);
fprintf('downloading image-domain learning network (8 coil) from %s\n', network_url) ;
websave(network_name, network_url);

% k-space deep learning w/ weighting network (1 coil)
network_path    = './network/cnn_residual_k_space_deep_learning_w_weight_init_8coil/';
network_name	= [network_path 'net-epoch-500.mat'];
network_url     = 'https://www.dropbox.com/s/ozwwuleegzbp61y/net-epoch-500.mat?dl=1';

mkdir(network_path);
fprintf('downloading k-space deep learning w/ weighitng network (8 coil) from %s\n', network_url) ;
websave(network_name, network_url);