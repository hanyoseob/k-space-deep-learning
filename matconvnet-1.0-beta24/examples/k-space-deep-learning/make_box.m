function box = make_box(iy, ix, h, w, sz)
    box                     = zeros(sz);
    
    box(iy, ix:ix+w-1)      = 1;
    box(iy+h-1, ix:ix+w-1)  = 1;
    box(iy:iy+h-1, ix)      = 1;
    box(iy:iy+h-1, ix+w-1)  = 1;
end