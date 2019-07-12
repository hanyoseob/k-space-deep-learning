
% --------------------------------------------------------------------
function [dst, nsmp]	= getBatchPatchVal(src, opts)
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

by      = (1:patch(1)) - 1;
bx      = (1:patch(2)) - 1;

%%
src     = padarray(src, [ker(1), ker(2)], 'both', 'symmetric');
dst    	= zeros(patch(1), patch(2), size(src, 3), nsmp, 'like', src);

for ix = 1:nix
    for iy = 1:niy
        iv              = niy*(ix - 1) + iy;
        dst(:,:,:,iv)	= src(iy_set(iy) + by,ix_set(ix) + bx,:);
    end
end

end
