% function sup    = SupportMask(sz)
% 
% r               = sz/2 + 1;
% [mx, my]        = meshgrid(linspace(-sz/2, sz/2, sz));
% 
% sup           	= sqrt(mx.^2 + my.^2) <= r;
% 
% end

function sup    = SupportMask(sz)

[mx, my]        = meshgrid(linspace(-sz(2), sz(2), sz(2)), linspace(-sz(1), sz(1), sz(1)));

sup           	= ((mx./sz(2)).^2 + (my./sz(1)).^2) <= 1 + 1e-2;

end