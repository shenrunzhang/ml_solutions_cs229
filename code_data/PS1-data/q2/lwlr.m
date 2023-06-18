function y = lwlr(X_train, y_train, x, tau)
% Output:
% Output y = 1{hθ(x) > 0.5} as the prediction.
%%% YOUR CODE HERE

lambda = 0.0001;
theta = [0 0]';

num_examples = size(y_train,1);

w = zeros(num_examples,1);

for i = 1:length(w)
    w(i) = exp(-(norm(X_train(i,:)' - x)^2) / (2 * tau^2));
end

function y = sig(x)
    y = 1 / (1 + exp(-x));
end

function zed = z(theta)
    zed = zeros(num_examples,1);
    for j = 1:length(w)
        zed(j) = w(j)*(y_train(j) - sig(theta' * X_train(j,:)'));
    end
end

function gradient = grad(theta)
    gradient = X_train' * z(theta)- lambda * theta;
end

function hess = hessian(theta)
    hess = X_train' * D(theta) * X_train - lambda * eye(2);
end

function d = D(theta)
    d = zeros(num_examples);
    for k = 1:num_examples
        d(k,k) = -w(k) * sig(theta' * X_train(k,:)') * (1 - sig(theta' * X_train(k,:)'));
    end
end

for i = 1:20
    theta = theta - hessian(theta) \ grad(theta);
end

% indicator on x > 0.5
if sig(theta' * x) > 0.5
    y = 1;
else
    y = 0;
end

end

