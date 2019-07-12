% function [W, R]   = WeightMask(nsz, sht, nnfft2, nnifft2, Rmax)
function [W, R]   = WeightMask(nsz, Rmax)

if nargin < 2
    Rmax = 1e-1;
end

% if nargin < 4
%     nnfft2  = @(x) fftshift(fftshift(fft2(x), 1), 2);
%     nnifft2 = @(x) ifft2(fftshift(fftshift(x, 1), 2));
% end
% 
% if nargin < 2
%     sht = ones(nsz);
% end
% 
% if length(nsz) == 2
%     nsz(3) = size(sht, 3);
%     nsz(4) = 1;
% end

ny      = nsz(1);
nx      = nsz(2);
% nch     = nsz(3);
% nbatch  = nsz(4);

if mod(nx,2)==0
    ix=-nx/2:nx/2-1;
else
    ix=-nx/2:nx/2;
end
if mod(ny,2)==0
    iy=-ny/2:ny/2-1;
else
    iy=-ny/2:ny/2;
end
wx          = Rmax*ix./(nx/2);
wy          = Rmax*iy./(ny/2);

[rwx,rwy]   = meshgrid(wx,wy);
R           = (rwx.^2+rwy.^2).^.5;

W           = single(R);

% W           = repmat(R, [1, 1, nch, nbatch]);
% if ~(sum(sht(:)) == ny*nx*nch)
%     W           = nnfft2(bsxfun(@times, nnifft2(W), sht));
% end
