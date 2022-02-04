function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 10;
sigma = 0.3;

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

C_col = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_col = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
n = size(C_col);
m = size(sigma_col);
error = 100;
for i = 1:n
    C_ = C_col(i);
    for j = 1:m
        sigma_ = sigma_col(j);
        model = svmTrain(X, y, C_, @(x1, x2)gaussianKernel(x1, x2, sigma_));
        predictions = svmPredict(model, Xval);
        error_ = mean(double(predictions ~= yval));
        if error_ < error
            C = C_;
            sigma = sigma_;
            error = error_;
        end
    end
end




% =========================================================================

end
