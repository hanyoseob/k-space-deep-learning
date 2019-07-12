classdef IFFT < dagnn.Filter
  properties
%     param = []
    nnfft2= @(x) fftshift(fftshift(fft2(ifftshift(ifftshift(x, 1), 2)), 1), 2);
    nnifft2= @(x) ifftshift(ifftshift(ifft2(fftshift(fftshift(x, 1), 2)), 1), 2);
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnifft(inputs{1}, self.nnfft2, self.nnifft2) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnifft(inputs{1}, self.nnfft2, self.nnifft2, derOutputs{1}) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = IFFT(varargin)
      obj.load(varargin) ;
    end
  end
end
