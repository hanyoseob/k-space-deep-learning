function y = vl_nnfft(x, nnfft2, nnifft2, dzdy)

if nargin == 3
%     if param.isreal
%         y = fftshift(fftshift(fft2(ifftshift(ifftshift(x, 1), 2)), 1), 2);
%     else
% %         y = fft2(complex(x(:,:,1:end/2,:), x(:,:,end/2+1:end,:)));
%         x = ri2comp(x);
% %         y = fftshift(fftshift(fft2(ifftshift(ifftshift(x, 1), 2)), 1), 2);
%         y = nnfft2(x);
%     end

    x = ri2comp(x);
%         y = fftshift(fftshift(fft2(ifftshift(ifftshift(x, 1), 2)), 1), 2);
    y = nnfft2(x);
    
%     y = cat(3, real(y), imag(y));

%     if isfield(param, 'support')
%         y	= bsxfun(@times, y, param.support);
%     end
    y   = comp2ri(y);
else
%     dzdy = complex(dzdy(:,:,1:end/2,:), dzdy(:,:,end/2+1:end,:));
%     y = ifft2(dzdy);
    dzdy	= ri2comp(dzdy);
    
%     if isfield(param, 'support')
%         dzdy    = bsxfun(@times, dzdy, param.support);
%     end
%     y       = ifftshift(ifftshift(ifft2(fftshift(fftshift(dzdy, 1), 2)), 1), 2);
    y   = nnifft2(dzdy);
    
%     if param.isreal
%         y = real(y);
%     else
% %         y = cat(3, real(y), imag(y));
%         y = comp2ri(y);
%     end
    y = comp2ri(y);
end