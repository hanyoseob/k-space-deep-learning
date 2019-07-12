% function net = add_block_multi(net, stgNum, filter, posStg, strBlk, opts)
function [net, lastLayer] = add_block_multi_img(net, nstg, filter, opts)

bBias       = opts.bBias;
bBnorm      = opts.bBnorm;
bReLU       = opts.bReLU;
nConnect	= opts.nConnect;

strScope    = opts.scope;

numStg      = opts.numStage;

cfy         = filter(1);
cfx         = filter(2);
cfz         = filter(3);
cfd         = filter(4);

hcfy        = floor((cfy - 1)/2);
hcfx        = floor((cfx - 1)/2);

hcf         = [hcfy, hcfy, hcfx, hcfx]; % [TOP BOTTOM LEFT RIGHT]

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONTRACTING PATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_con_conv1 	= dagnn.Conv('size', [cfy, cfx, cfz, cfd], 'pad', hcf, 'stride', 1, 'hasBias', bBias);
l_con_bnorm1    = dagnn.BatchNorm('numChannels', cfd, 'epsilon', 1e-5);
l_con_relu1     = dagnn.ReLU();

l_con_conv2 	= dagnn.Conv('size', [cfy, cfx, cfd, cfd], 'pad', hcf, 'stride', 1, 'hasBias', bBias);
l_con_bnorm2    = dagnn.BatchNorm('numChannels', cfd, 'epsilon', 1e-5);
l_con_relu2     = dagnn.ReLU();

% l_con_mp        = dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'pad', 0, 'stride', 2);
l_con_mp        = dagnn.Pooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BI-PASSING PATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_bi_conv1      = dagnn.Conv('size', [cfy, cfx, cfd, 2*cfd], 'pad', hcf, 'stride', 1, 'hasBias', bBias);
l_bi_bnorm1     = dagnn.BatchNorm('numChannels', 2*cfd, 'epsilon', 1e-5);
l_bi_relu1      = dagnn.ReLU();

l_bi_conv2      = dagnn.Conv('size', [cfy, cfx, 2*cfd, cfd], 'pad', hcf, 'stride', 1, 'hasBias', bBias);
l_bi_bnorm2     = dagnn.BatchNorm('numChannels', cfd, 'epsilon', 1e-5);
l_bi_relu2      = dagnn.ReLU();

% l_bi_convt      = dagnn.ConvTranspose('size', [2, 2, cfd, cfd], 'crop', 0, 'upsample', 2, 'hasBias', true);
l_bi_convt    	= dagnn.UnPooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPANSIVE PATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch nConnect
    case 1
        l_ext_connect	= dagnn.Sum();
    case 2
        l_ext_connect	= dagnn.Concat();
end

l_ext_conv1 	= dagnn.Conv('size', [cfy, cfx, nConnect*cfd, cfd], 'pad', hcf, 'stride', 1, 'hasBias', bBias);
l_ext_bnorm1    = dagnn.BatchNorm('numChannels', cfd, 'epsilon', 1e-5);
l_ext_relu1     = dagnn.ReLU();

l_ext_conv2 	= dagnn.Conv('size', [cfy, cfx, cfd, cfd/(2^boolean(nstg))], 'pad', hcf, 'stride', 1, 'hasBias', bBias);
l_ext_bnorm2    = dagnn.BatchNorm('numChannels', cfd/(2^boolean(nstg)), 'epsilon', 1e-5);
l_ext_relu2 	= dagnn.ReLU();

% l_ext_convt     = dagnn.ConvTranspose('size', [2, 2, cfd, cfd], 'crop', 0, 'upsample', 2, 'hasBias', true);
l_ext_convt    	= dagnn.UnPooling('method', 'avg', 'poolSize', [2, 2], 'pad', 0, 'stride', [2, 2]);

% l_ext_scale    	= dagnn.Scale('hasBias', false, 'scale', 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONTRACTING PATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nameCon_pre     = ['stg' num2str(nstg - 1) '_con' strScope];
nameCon      	= ['stg' num2str(nstg + 0) '_con' strScope];

if nstg
    lastLayer   = [nameCon_pre '_mp'];
else
    lastLayer   = opts.input;
end

net.addLayer(['l_' nameCon '_conv1'],	l_con_conv1,	{lastLayer},            {[nameCon '_conv1']},	{[nameCon '_c1f'],  [nameCon '_c1b']});
net.addLayer(['l_' nameCon '_bnorm1'],	l_con_bnorm1,	{[nameCon '_conv1']},   {[nameCon '_bnorm1']},	{[nameCon '_bn1f'], [nameCon '_bn1b'], [nameCon '_bn1m']});
net.addLayer(['l_' nameCon '_relu1'],   l_con_relu1,	{[nameCon '_bnorm1']},  {[nameCon '_relu1']});

net.addLayer(['l_' nameCon '_conv2'],   l_con_conv2,    {[nameCon '_relu1']}, 	{[nameCon '_conv2']},	{[nameCon '_c2f'],  [nameCon '_c2b']});
net.addLayer(['l_' nameCon '_bnorm2'],  l_con_bnorm2,	{[nameCon '_conv2']},	{[nameCon '_bnorm2']},	{[nameCon '_bn2f'], [nameCon '_bn2b'], [nameCon '_bn2m']});
net.addLayer(['l_' nameCon '_relu2'],   l_con_relu2,	{[nameCon '_bnorm2']},	{[nameCon '_relu2']});

lastLayer   = [nameCon '_relu2'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BI-PASS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nameBi              = ['stg' num2str(nstg + 0) '_bi'];

nameExt             = ['stg' num2str(nstg + 0) '_ext' strScope];
nameExt_next        = ['stg' num2str(nstg + 1) '_ext' strScope];

if ((numStg - 1) == nstg)    
    net.addLayer(['l_' nameCon '_mp'],          l_con_mp,       {lastLayer},                    {[nameCon '_mp']});
    
    net.addLayer(['l_' nameExt_next '_conv1'],	l_bi_conv1,     {[nameCon '_mp']},              {[nameExt_next '_conv1']},	{[nameExt_next '_c1f'],     [nameExt_next '_c1b']});
    net.addLayer(['l_' nameExt_next '_bnorm1'],	l_bi_bnorm1,	{[nameExt_next '_conv1']},      {[nameExt_next '_bnorm1']},	{[nameExt_next '_bn1f'],    [nameExt_next '_bn1b'],     [nameExt_next '_bn1m']});
    net.addLayer(['l_' nameExt_next '_relu1'],	l_bi_relu1,     {[nameExt_next '_bnorm1']},     {[nameExt_next '_relu1']});
    
    net.addLayer(['l_' nameExt_next '_conv2'],	l_bi_conv2,     {[nameExt_next '_relu1']},  	{[nameExt_next '_conv2']},	{[nameExt_next '_c2f'],     [nameExt_next '_c2b']});
    net.addLayer(['l_' nameExt_next '_bnorm2'],	l_bi_bnorm2,	{[nameExt_next '_conv2']},      {[nameExt_next '_bnorm2']},	{[nameExt_next '_bn2f'],	[nameExt_next '_bn2b'],     [nameExt_next '_bn2m']});
    net.addLayer(['l_' nameExt_next '_relu2'],	l_bi_relu2,     {[nameExt_next '_bnorm2']},     {[nameExt_next '_relu2']});    
    
%     net.addLayer(['l_' nameExt '_convt'],       l_bi_convt,     {[nameExt_next '_relu2']},  	{[nameExt '_convt']},       {[nameExt '_ct2f'],         [nameExt '_ct2b']});
    net.addLayer(['l_' nameExt '_convt'],       l_bi_convt,     {[nameExt_next '_relu2']},  	{[nameExt '_convt']});
else
    net.addLayer(['l_' nameCon '_mp'],          l_con_mp,       {lastLayer},                    {[nameCon '_mp']});
%     net.addLayer(['l_' nameExt '_convt'],       l_ext_convt,	{[nameExt_next '_relu2']},      {[nameExt '_convt']},       {[nameExt '_ct2f'],         [nameExt '_ct2b']});
    net.addLayer(['l_' nameExt '_convt'],       l_ext_convt,	{[nameExt_next '_relu2']},      {[nameExt '_convt']});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPANSIVE PATH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nameExt             = ['stg' num2str(nstg + 0) '_ext'];
% net.addLayer(['l_' nameExt '_scale'],  	l_ext_scale,        {lastLayer},            {[nameExt '_scale']},   {[nameExt '_s1']});
net.addLayer(['l_' nameExt '_connect'],	l_ext_connect,      {lastLayer, [nameExt '_convt']},	{[nameExt '_connect']});

net.addLayer(['l_' nameExt '_conv1'],   l_ext_conv1,        {[nameExt '_connect']},	{[nameExt '_conv1']},	{[nameExt '_c1f'],  [nameExt '_c1b']});
net.addLayer(['l_' nameExt '_bnorm1'],	l_ext_bnorm1,       {[nameExt '_conv1']},	{[nameExt '_bnorm1']},	{[nameExt '_bn1f'], [nameExt '_bn1b'], [nameExt '_bn1m']});
net.addLayer(['l_' nameExt '_relu1'], 	l_ext_relu1,        {[nameExt '_bnorm1']},	{[nameExt '_relu1']});

net.addLayer(['l_' nameExt '_conv2'],	l_ext_conv2,        {[nameExt '_relu1']}, 	{[nameExt '_conv2']},	{[nameExt '_c2f'],  [nameExt '_c2b']});
net.addLayer(['l_' nameExt '_bnorm2'],	l_ext_bnorm2,       {[nameExt '_conv2']},	{[nameExt '_bnorm2']},	{[nameExt '_bn2f'], [nameExt '_bn2b'], [nameExt '_bn2m']});
net.addLayer(['l_' nameExt '_relu2'], 	l_ext_relu2,        {[nameExt '_bnorm2']},	{[nameExt '_relu2']});

lastLayer   = [nameExt '_relu2'];
end
