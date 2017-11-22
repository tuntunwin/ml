function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;

diff = h - y;
Ju = 1/(2*m) * (diff' * diff);
r = lambda/(2 * m) * sum(theta(2:end,:) .^2);
J = Ju + r;


grad_u = 1/m * ((h - y)' * X)';
grad_r =[0;[(lambda / m) * theta(2:end,:)]];
grad = grad_u + grad_r;




% =========================================================================

grad = grad(:);

end
