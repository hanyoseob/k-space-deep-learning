classdef SubPixelT < dagnn.ElementWise
  properties
    poolSize = [1 1]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnsubpixelt(inputs{1}, self.poolSize) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnsubpixelt(inputs{1}, self.poolSize, derOutputs{1}) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = SubPixelT(varargin)
      obj.load(varargin) ;
    end
  end
end
