classdef UnWeighting < dagnn.ElementWise
  properties
    wgt = [];
  end

  methods

    function outputs = forward(obj, inputs, params)
%       args = horzcat(inputs, params) ;
%       outputs{1} = bsxfun(@times, args{1}, args{2}) ;
%       if obj.hasBias
%         outputs{1} = bsxfun(@plus, outputs{1}, args{3}) ;
%       end
%       outputs{1}                = inputs{1}./obj.w(:,:,:,1:size(inputs{1},4));

%       inputs_   = complex(inputs{1}(:,:,1:end/2,:), inputs{1}(:,:,end/2+1:end,:));
%       outputs_	= wgt2k(inputs_, inputs{2}, inputs{3});
%       outputs{1}= cat(3, real(outputs_), imag(outputs_));

      inputs_       = ri2comp(inputs{1});
      dc_           = inputs{2};
      wgt_          = obj.wgt;
      outputs_      = wgt2k(inputs_, wgt_, dc_);
      outputs{1}    = comp2ri(outputs_);
      
      
%       outputs{1}                = bsxfun(@times, inputs{1}, 1./inputs{2});
%       outputs{1}(inputs{2} == 0)= inputs{3};
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
        
%       derOutputs_	= complex(derOutputs{1}(:,:,1:end/2,:), derOutputs{1}(:,:,end/2+1:end,:));
%       derInputs_    = k2wgt(derOutputs_);
%       derInputs{1}	= cat(3, real(derInputs_), imag(derInputs_));
%       derInputs{1}	= bsxfun(@times, derOutputs{1}, 1./obj.w(:,:,1,1));

      derOutputs_	= ri2comp(derOutputs{1});
      dc_           = inputs{2};
      wgt_          = obj.wgt;
      derInputs_    = k2wgt(derOutputs_, wgt_, dc_);
      derInputs{1}	= comp2ri(derInputs_);
      derInputs{2}  = zeros(size(dc_), 'like', dc_);
      
%       derInputs{2}	= [];
%       derInputs{3}	= [];
%       derInputs{1}          	= derOutputs{1}./obj.w(:,:,:,1:size(inputs{1},4));
%       outputs{1}(inputs{2} == 0)= 0;
      derParams                 = {} ;
    end

    function obj = UnWeighting(varargin)
      obj.load(varargin) ;
    end
  end
end
