function y = vl_nnifft(x, nnfft2, nnifft2, dzdy)

if nargin == 3
    
    %     y = ifftshift(ifftshift(ifft2(fftshift(fftshift(complex(x(:,:,1:end/2,:), x(:,:,end/2+1:end,:)), 1), 2)), 1), 2);
    x = ri2comp(x);
%     if isfield(param, 'support')
%         x = bsxfun(@times, x, param.support);
%     end
%     y = ifftshift(ifftshift(ifft2(fftshift(fftshift(x, 1), 2)), 1), 2);
    y = nnifft2(x);
    
%     if param.isreal
%         y       = real(y);
%     else
%         %         y = cat(3, real(y), imag(y));
%         y       = comp2ri(y);
%     end
    
    y       = comp2ri(y);
    
else
%     if param.isreal
% %         y       = fftshift(fftshift(fft2(ifftshift(ifftshift(dzdy, 1), 2)), 1), 2);
%         y       = nnfft2(dzdy);
%     else
%         %         y = fftshift(fftshift(fft2(ifftshift(ifftshift(complex(dzdy(:,:,1:end/2,:), dzdy(:,:,end/2+1:end,:)), 1), 2)), 1), 2);
%         dzdy    = ri2comp(dzdy);
% %         y       = fftshift(fftshift(fft2(ifftshift(ifftshift(dzdy, 1), 2)), 1), 2);
%         y       = nnfft2(dzdy);
%     end

    dzdy    = ri2comp(dzdy);
    %         y       = fftshift(fftshift(fft2(ifftshift(ifftshift(dzdy, 1), 2)), 1), 2);
    y       = nnfft2(dzdy);
    
    %     y = cat(3, real(y), imag(y));
%     if isfield(param, 'support')
%         y	= bsxfun(@times, y, param.support);
%     end
    y   = comp2ri(y);
end