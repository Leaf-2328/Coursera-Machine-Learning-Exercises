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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% first method
a1 = [ones(m,1) X]; % 5000*401
z2 = Theta1*a1'; % 25*5000
a2 = [ones(size(sigmoid(z2)',1),1) sigmoid(z2)']; % 5000*26
z3 = Theta2*a2'; % 10*5000
h_x = sigmoid(z3)'; % 5000*10

y_vector = zeros(m,num_labels); % 5000*10
for i = 1:m
    y_vector(i,y(i)) = 1;
end
J = (1/m) * sum(sum((-y_vector.*log(h_x))-((1-y_vector).*log(1-h_x))));  %scalar

%regularizing
t1 = Theta1(:,2:end);
t2 = Theta2(:,2:end);
J = J + (lambda/(2*m))*(sum(sum(t1.*t1))+sum(sum(t2.*t2)));

% % error:
% % y_vector = zeros(m,num_labels); % 5000*10
% % for i = 1:m
% %     y_vector(i,y(i)) = 1;
% % end
% % 
% % sum_total = 0;
% % for row = 1:m
% %     a1 = [1 X(m,:)]; %1*401
% %     y_ = y_vector(m,:); % 1*10
% %     % Theta1 25*401
% %     % Theta2 10*26
% %     z2 = Theta1*a1'; % 25*1` 
% %     a2 = [1 sigmoid(z2)']; %1*26
% %     z3 = Theta2*a2'; % 10*1
% %     h = sigmoid(z3)'; % 1*10
% %     sum_1 = sum((-y_.*log(h))-((1-y_).*log(1-h)));
% %     sum_total = sum_total + sum_1;
% % end
% % 
% % J = (1/m)*sum_total;


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

% backprop #1
for t = 1:m
    x_t = X(t,:);
    a1 = [1 x_t]; % 1*401
    z2 = Theta1*a1'; % 25*1
    a2 = [1 sigmoid(z2)']; % 1*26
    z3 = Theta2*a2'; % 10*1
    a3 = sigmoid(z3); % 10*1
    
    delta_3 = a3- y_vector(t,:)'; % 10*1
    delta_2 = (Theta2'*delta_3).*[1; sigmoidGradient(z2)]; % 26*1
    delta_2 = delta_2(2:end); % 25*1
    % Theta1 25*401
    % Theta2 10*26
    Theta1_grad = Theta1_grad + (delta_2*a1);
    Theta2_grad = Theta2_grad + (delta_3*a2);
end

Theta1_grad = (1/m)*Theta1_grad+[zeros(size(Theta1,1),1) (lambda/m)*Theta1(:,2:end)];
Theta2_grad = (1/m)*Theta2_grad+[zeros(size(Theta2,1),1) (lambda/m)*Theta2(:,2:end)];

% backprop #2
% X = [ones(m,1) X];
% for t=1:m
%     % Here X is including 1 column at begining
%     
%     % for layer-1
%     a1 = X(t,:)'; % (n+1) x 1 == 401 x 1
%     
%     % for layer-2
%     z2 = Theta1 * a1;  % hidden_layer_size x 1 == 25 x 1
%     a2 = [1; sigmoid(z2)]; % (hidden_layer_size+1) x 1 == 26 x 1
%   
%     % for layer-3
%     z3 = Theta2 * a2; % num_labels x 1 == 10 x 1    
%     a3 = sigmoid(z3); % num_labels x 1 == 10 x 1    
% 
%     yVector = (1:num_labels)'==y(t); % num_labels x 1 == 10 x 1    
%     
%     %calculating delta values
%     delta3 = a3 - yVector; % num_labels x 1 == 10 x 1    
%     
%     delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)]; % (hidden_layer_size+1) x 1 == 26 x 1
%     
%     delta2 = delta2(2:end); % hidden_layer_size x 1 == 25 x 1 %Removing delta2 for bias node  
%     
%     % delta_1 is not calculated because we do not associate error with the input  
%     
%     % CAPITAL delta update
%     Theta1_grad = Theta1_grad + (delta2 * a1'); % 25 x 401
%     Theta2_grad = Theta2_grad + (delta3 * a2'); % 10 x 26
%  
% end
% 
% Theta1_grad = (1/m) * Theta1_grad; % 25 x 401
% Theta2_grad = (1/m) * Theta2_grad; % 10 x 26



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


end
