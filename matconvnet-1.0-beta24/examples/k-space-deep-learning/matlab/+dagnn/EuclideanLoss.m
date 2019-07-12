%PDIST vl_nnpdist dagnn wrapper
%  Accepts 2 or 3 inputs, where third input is used as variable
%  'instanceWeights' parameter. Derivatives for the 3rd input are not
%  computed.
%  By default aggregates the element-wise loss.
classdef EuclideanLoss < dagnn.Loss
  properties
    p           = 2;
    aggregate	= true;
    wgt         = 1;
  end

  methods
    function outputs = forward(obj, inputs, params)
      switch numel(inputs)
        case 2
          outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, [], ...
            'aggregate', true, 'noRoot', true, obj.opts{:}) ;
        case 3
          outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, [], ...
            'aggregate', true, 'noRoot', true, 'instanceWeights', inputs{3}, ...
            obj.opts{:}) ;
        otherwise
          error('Invalid number of inputs');
      end
      outputs{1} = outputs{1}.*obj.wgt;
      obj.accumulateAverage(inputs, outputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1, numel(inputs));
      switch numel(inputs)
        case 2
          [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, ...
            obj.p, derOutputs{1}, 'aggregate', false, 'noRoot', true, obj.opts{:}) ;
        case 3
          [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, ...
            obj.p, derOutputs{1}, 'aggregate', false, 'noRoot', true, ...
            'instanceWeights', inputs{3}, obj.opts{:}) ;
        otherwise
          error('Invalid number of inputs');
      end
      derParams = {} ;
      
      derInputs{1}  = derInputs{1}./(size(derInputs{1}, 1) * size(derInputs{1}, 2) * size(derInputs{1}, 3)).*obj.wgt;
      derInputs{2}  = derInputs{2}./(size(derInputs{2}, 1) * size(derInputs{2}, 2) * size(derInputs{2}, 3)).*obj.wgt;
    end

    function obj = EuclideanLoss(varargin)
      obj.load(varargin) ;
      obj.loss = 'pdist';
    end
  end
end
