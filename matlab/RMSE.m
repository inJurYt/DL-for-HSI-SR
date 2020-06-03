function [RMSE] = RMSE(recon,truth)
% RMSE in 8-bit range [0...255]

diff = double(truth)-double(recon);

MSE = sum(sum(diff.^2)) /size(truth,1) /size(truth,2);

RMSE = sqrt( MSE );

% These are equivalent defintions
%
% RMSE = sqrt( norm(diff,'fro')^2 /size(truth,1) /size(truth,2) );
% 
% RMSE =  norm(diff,'fro') * sqrt(1 /size(truth,1) /size(truth,2) );

