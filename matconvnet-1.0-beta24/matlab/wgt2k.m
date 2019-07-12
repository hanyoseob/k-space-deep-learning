function X	= wgt2k(Y, W, DC)

X                   = bsxfun(@times, Y, 1./W);

[idcy, idcx]        = find(W == 0);

if ~isempty(idcy)
    X(idcy, idcx, :, :)	= DC;
end
