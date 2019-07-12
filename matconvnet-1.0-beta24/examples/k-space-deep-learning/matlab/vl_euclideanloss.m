function y = vl_euclideanloss(x, c, dzdy)
%EUCLIDEANLOSS Summary of this function goes here
%   Detailed explanation goes here

assert(numel(x) == numel(c));

d = size(x);

assert(all(d == size(c)));

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    
    y = 1 / 2 / prod(d) * sum(subsref((x - c) .^ 2, substruct('()', {':'}))); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = 1 / (2 * prod(d(1 : 3))) * sum(subsref((X - c) .^ 2, substruct('()', {':'}))); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    y = dzdy / prod(d) * (x - c); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = dzdy / prod(d(1 : 3)) * (X - c); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
end

end
