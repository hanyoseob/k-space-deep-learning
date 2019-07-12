classdef Weighting < dagnn.ElementWise
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
%       outputs{1}                = inputs{1}.*inputs{2};
%       outputs{1}(inputs{2} == 0)= inputs{3};

      inputs_       = ri2comp(inputs{1});
      dc_           = inputs{2};
      wgt_         	= obj.wgt;
      
      outputs_      = k2wgt(inputs_, wgt_, dc_);
      outputs{1}    = comp2ri(outputs_);
%       outputs{2}            = w_;
%       outputs{3}            = dc_;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%       derInputs{1}          	= derOutputs{1}.*inputs{2};
%       derParams                 = {} ;
      derOutputs_  	= ri2comp(derOutputs{1});
      dc_           = inputs{2};
      wgt_         	= obj.wgt;
      derInputs_   	= wgt2k(derOutputs_, wgt_, dc_);
      derInputs{1}  = comp2ri(derInputs_);
      derInputs{2}  = zeros(size(dc_), 'like', dc_);
      
%       derInputs{2}          = inputs{2};
%       derInputs{3}          = inputs{3};
      
      derParams             = {} ;
    end

    function obj = Weighting(varargin)
      obj.load(varargin) ;
    end
  end
end
