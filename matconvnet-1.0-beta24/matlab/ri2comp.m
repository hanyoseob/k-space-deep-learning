function dst    = ri2comp(src)
dst   = complex(src(:,:,1:end/2,:), src(:,:,end/2+1:end,:));
end