function rec = recon_cnn4img_comp(net, data, opts)

if ~isempty(opts.gpus)
    net.move('gpu');
end

net.vars(opts.vid).precious	= true ;

rec     = zeros(opts.size, 'single');
data    = data + opts.offset;
set     = opts.set;

for ival	= 1:1:length(set)
    
    iz          = set(ival);
    
%     disp([num2str(iz) ' / ' num2str(set(end))]);
    
    data_       = single(squeeze(data(:,:,:,iz)));
    
    data_patch	= getBatchPatchVal(data_, opts);
    
    if opts.meanNorm
        means_patch_  	= mean(mean(mean(data_patch, 1), 2), 3);
    else
        means_patch_ 	= 0;
    end
    
    data_patch	= bsxfun(@minus, data_patch, means_patch_);
    
    if opts.varNorm
        vars_patch      = max(max(max(abs(data_patch), [], 1), [], 2), [], 3);
    else
        vars_patch      = 1;
    end
    
    data_patch	= bsxfun(@times, opts.wgt*data_patch, 1./vars_patch);
    
    
    if opts.isweight
        [idcy, idcx]    = find(opts.weight(:,:,1) == 0);
        data_patch_ft   = opts.nnfft2(data_patch);
        dc_fft_patch	= data_patch_ft(idcy, idcx, :, :);
    end
    
    data_patch  = comp2ri(data_patch);
    
    %%
    nbatch      = size(data_patch, 4);
    batch_      = (1:opts.batchSize) - 1;
    
    rec_batch = single([]);
    
    for ibatch  = 1:opts.batchSize:nbatch
        batch                       = ibatch + batch_;
        batch(batch > nbatch)       = [];
        
        data_batch                  = data_patch(:,:,:,batch);
        
        if ~isempty(opts.gpus)
            data_batch	= gpuArray(data_batch);
        end
        
        if opts.isweight
            dc_batch 	= dc_fft_patch(:,:,:,batch);
            net.eval({opts.input;data_batch;'dc_fft';dc_batch}) ;
        else
            net.eval({opts.input;data_batch}) ;
        end

        rec_batch_                  = net.vars(opts.vid).value;

        rec_batch(:,:,:,batch)      = gather(rec_batch_);
    end
    
    rec_batch       = bsxfun(@times, rec_batch/opts.wgt, vars_patch);
    
    rec_batch       = ri2comp(rec_batch);
    
    rec_            = getReconPatchVal(rec_batch, opts);
    rec(:,:,:,ival)	= bsxfun(@plus, rec_, means_patch_);
    
end

rec             = rec - opts.offset;

net.reset();
net.move('cpu');

end