classdef UnPooling < dagnn.Filter
  properties
    method = 'avg'
    poolSize = [1 1]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnpool(repmat(inputs{1}, self.poolSize), self.poolSize, inputs{1}, ...
                             'pad', self.pad, ...
                             'stride', self.stride, ...
                             'method', self.method, ...
                             self.opts{:}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnpool(derOutputs{1}, self.poolSize, ...
                               'pad', self.pad, ...
                               'stride', self.stride, ...
                               'method', self.method, ...
                               self.opts{:}) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = UnPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
