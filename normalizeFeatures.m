%% ******************* Features Normalization *******************
function norm_X = normalizeFeatures(X)
norm_X = X;% contains normalized features
mu = zeros(1,size(X,2));% includes the mean of each feature
sd = zeros(1,size(X,2));% includes the standard deviation of each feature

    for i = 1:size(X,2)
        mu(i) = mean(X(:,i));
        sd(i) = std(X(:,i));
        norm_X(:,i) = (norm_X(:,i) - (mu(i)*ones(size(X, 1),1)))./sd(i);
    end

disp('The features were normalized.');
end

%% This is vectorized implementation, you can ignore it :)
% function [X_norm, mu, sigma] = normalizeFeatures(X)
% %   Normalizes the features in X
% %   It returns a normalized version of X where the mean value of each feature 
% %   is 0 and the standard deviation is 1. 
% %   This is often a good preprocessing step to do when working with learning algorithms.
% 
%   % You need to set these values correctly
%   X_norm = X;
%   mu = zeros(1, size(X, 2));
%   sigma = zeros(1, size(X, 2));
% 
%   mu = mean(X);
%   sigma = std(X);
%   X_norm = (X - repmat(mu, size(X,1),1)) ./ repmat(sigma, size(X,1),1);
% 
% end
