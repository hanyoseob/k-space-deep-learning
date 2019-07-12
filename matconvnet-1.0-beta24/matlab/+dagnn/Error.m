%PDIST vl_nnpdist dagnn wrapper
%  Accepts 2 or 3 inputs, where third input is used as variable
%  'instanceWeights' parameter. Derivatives for the 3rd input are not
%  computed.
%  By default aggregates the element-wise loss.
classdef Error < dagnn.Loss
    properties
        %     loss	= 'psnr'
        %     ignoreAverage   = true
        method = 'image'
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            
%             switch numel(inputs)
%                 case 4
%                     inputs{1}   = inputs{4} - inputs{1} ;
%                     inputs{2}   = inputs{4} - inputs{2} ;
%             end
%             
%             
%             inputs{1}   = bsxfun(@plus, inputs{1}, inputs{3});
%             inputs{2}   = bsxfun(@plus, inputs{2}, inputs{3});
            
            peakval     = max(max(max(abs(inputs{2}),[],1),[],2),[],3);
            
            peakval(peakval == 0) = 1;
            
            inputs{1}   = bsxfun(@times, inputs{1}, 1./peakval);
            inputs{2}   = bsxfun(@times, inputs{2}, 1./peakval);
            
            switch obj.loss
                case 'psnr'
                    outputs{1} = psnr(gather(inputs{2}(:)), gather(inputs{1}(:))) ;
                case 'mse'
                    outputs{1} = immse(gather(inputs{2}(:)), gather(inputs{1}(:))) ;
                case 'l2'
                    outputs{1} = norm(gather(inputs{2}(:)) - gather(inputs{1}(:))) ;
                otherwise
                    error('Invalid number of inputs');
            end
            obj.accumulateAverage(inputs, outputs);
        end
        
        function accumulateAverage(obj, inputs, outputs)
            if obj.ignoreAverage, return; end;
            n = obj.numAveraged ;
            m = n + size(inputs{1}, 4);
            obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1}*size(inputs{1}, 4))) / m ;
            obj.numAveraged = m ;
        end
        
        function obj = Error(varargin)
            obj.load(varargin) ;
            obj.method  = obj.method;
        end
    end
end
