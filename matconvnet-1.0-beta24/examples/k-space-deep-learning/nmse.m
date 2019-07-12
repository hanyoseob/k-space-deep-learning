function [ nmse ] = nmse( result, ground_truth)
%NMSE 이 함수의 요약 설명 위치
%   자세한 설명 위치
nmse   =( norm( result(:) - ground_truth(:) )/ norm( ground_truth(:) ) )^2;

end

