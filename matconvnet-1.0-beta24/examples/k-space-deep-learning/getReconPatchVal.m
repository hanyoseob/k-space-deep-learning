
% --------------------------------------------------------------------
function [dst, wgt] = getReconPatchVal(src, opts)
% --------------------------------------------------------------------
nsz     = opts.imageSize;
patch   = opts.inputSize;
ker     = opts.kernalSize;

nsz(1)	= nsz(1) + 2*ker(1);
nsz(2)	= nsz(2) + 2*ker(2);

niy     = ceil(nsz(1)/patch(1) + nsz(1)/patch(1)*2*ker(1)/patch(1));
nix     = ceil(nsz(2)/patch(2) + nsz(2)/patch(2)*2*ker(2)/patch(2));
nsmp    = niy*nix;

%%
iy_set  = fix(linspace(1, nsz(1) - patch(1) + 1, niy));
ix_set	= fix(linspace(1, nsz(2) - patch(2) + 1, nix));

% by      = (1:patch(1)) - 1;
% bx      = (1:patch(2)) - 1;

by      = (ker(1)+1:patch(1)-ker(1)) - 1;
bx      = (ker(2)+1:patch(2)-ker(2)) - 1;

%%
dst    	= zeros(nsz(1), nsz(2), nsz(3), 'like', src);
wgt     = zeros(nsz(1), nsz(2), nsz(3), 'like', src);

for ix = 1:nix
    for iy = 1:niy
        iv                                      = niy*(ix - 1) + iy ;
        
        dst(iy_set(iy) + by,ix_set(ix) + bx,:)  = dst(iy_set(iy) + by,ix_set(ix) + bx,:) + src(by + 1,bx + 1,:,iv);
        wgt(iy_set(iy) + by,ix_set(ix) + bx,:)	= wgt(iy_set(iy) + by,ix_set(ix) + bx,:) + 1 ;
        
    end
end

dst     = dst(ker(1)+1:end-ker(1), ker(2)+1:end-ker(2), :);
wgt     = wgt(ker(1)+1:end-ker(1), ker(2)+1:end-ker(2), :);

dst     = dst./wgt;
