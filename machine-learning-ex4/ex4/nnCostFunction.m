function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%disp('nn_params'),size(nn_params)
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
y_v = full(sparse(1:numel(y), y, 1));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% size(y)

a1_b = [ones(m,1) X];
%disp('a1'),size(a1)

z2 = Theta1 * a1_b';
%disp('z2'),size(z2)

a2 = sigmoid(z2);
a2_b = [ones(1,size(a2,2)); a2];
z3 = Theta2 * a2_b;
a3 = sigmoid(z3);
a3_t = a3';
diff = (-y_v .* log(a3_t)) - ((1-y_v) .* log(1 - a3_t));

r = lambda /(2 * m) * (sum((Theta1(:,2:end).^2)(:)) + sum((Theta2(:,2:end) .^2)(:)));
J = (1/m * sum((diff)(:))) +  r;

% Back Propagation

d_3 = a3_t - y_v;
%disp('d_3'),size(d_3)
z2_b = [ones(1, size(z2,2));z2];
%disp('z2_b'),size(z2_b)

g2_p = sigmoidGradient(z2_b);
%disp('g2_p'),size(g2_p)

p_2 = Theta2' * d_3';
%disp('p_2'),size(p_2)

% disp('a'),size(a2)
% disp('d_3'),size(d_3)
d_2 = p_2 .* g2_p; 
d_2_u = d_2(2:end,:);
dd_2 = d_3' * a2_b';
dd_1 = d_2_u * a1_b;

r1 = (lambda / m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)];
r2 = (lambda / m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)];

%rr1 = (lambda / m) * [zeros(1, size(r1,2));r1(2:end,:)];
%rr2 = (lambda / m) * [zeros(1, size(r2,2));r2(2:end,:)];

ddd_1 = 1/m * dd_1;
ddd_2 = 1/m * dd_2;

%disp('Theta1'),size(Theta1)
%disp('Theta2'),size(Theta2)

% disp('Theta2'),size(Theta2)
%disp('dd_1'),size(dd_1)
%disp('dd_2'),size(dd_2)
Theta1_grad = ddd_1 + r1;
Theta2_grad = ddd_2 + r2;

%J = sum(J_u + J_r);
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%disp('grad'),size(grad)
%disp('nn_params'),size(nn_params)
end
