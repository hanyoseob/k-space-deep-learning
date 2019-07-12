function res = ssos(x ,dim, pnorm)
% res = sos(x [,dim, pnorm])
%
% function computes the square root of sum of squares along dimension dim.
% If dim is not specified, it computes it along the last dimension.
%
% (c) Michael Lustig 2009
if length(size(x))==2
    res = abs(x);
else
    if nargin < 2
        dim = size(size(x),2);
    end
    
    if nargin < 3
        pnorm = 2;
    end
    
    
    % res = (sum(abs(x.^pnorm),dim)).^(1/pnorm);
    res = squeeze(sum(abs((x./size(x,dim)).^pnorm),dim)).^(1/pnorm); % edited by DW 20160317
end