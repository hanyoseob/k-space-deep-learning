function epoch = findLastCheckpoint(modelDir, modelName)
% -------------------------------------------------------------------------
if nargin < 2
    modelName = 'net_epoch';
end

list = dir(fullfile(modelDir, [modelName '-*.mat'])) ;
tokens = regexp({list.name}, [modelName '-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
