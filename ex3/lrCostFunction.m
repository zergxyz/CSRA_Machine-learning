function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

hx = sigmoid(X * theta);
J = 1 / m * ( -1 * y' * log(hx) - (1-y') * log(1 - hx) );
grad = 1 / m * (X' * (hx - y));  

theta_zeroed_first = [0; theta(2:length(theta));];
J = J+lambda/(2*m)*sum(theta_zeroed_first .^2);
grad = grad+lambda/m*theta_zeroed_first;

grad = grad(:);

end
