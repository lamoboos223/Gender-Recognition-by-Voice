function [J, grad] = calculateCost(X, y, theta, lambda)
%calculate cost with regularization 
m = length(y);
theta2 = [0 ; theta(2:end, :)];
h = sigmoid(X*theta);

J = (1/m) * (-y' * log(h) - (1 - y)' * log(1 - h)) +  lambda / (2*m) * (theta2' * theta2);
grad = (1/m) * (X' * (h - y)) + lambda / m * theta .* [0; ones(size(theta, 1) - 1, 1)];

end
