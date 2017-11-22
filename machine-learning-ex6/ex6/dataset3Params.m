function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
C = 1;
sigma = 0.1;
%{
% You need to return the following variables correctly.
test = [0.01 0.03 0.1 0.3 1 3 10 30];
count = length(test);
errors = zeros(count^2,3);


  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Fill in this function to return the optimal C and sigma
  %               learning parameters found using the cross validation set.
  %               You can use svmPredict to predict the labels on the cross
  %               validation set. For example, 
  %                   predictions = svmPredict(model, Xval);
  %               will return the predictions on the cross validation set.
  %
  %  Note: You can compute the prediction error using 
  %        mean(double(predictions ~= yval))
  %
%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
for i=1:count
  for j=1:count
    C_t = test(i);
    sigma_t = test(j);
    model= svmTrain(X, y, C_t, @(x1, x2) gaussianKernel(x1, x2, sigma_t));
    predictions = svmPredict(model, Xval);
    ix = ((i-1) * count) + j;
    errors(ix,:) = [C_t, sigma_t, mean(double(predictions ~= yval))];
    ix,errors(ix,:)
  end
end
%erros
[v i] = min(errors(:,3))
C = errors(i,1)
sigma = errors(i, 2)

% =========================================================================
%}
end
