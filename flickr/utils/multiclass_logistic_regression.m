%%% multiclass logistic regression
%   x                   : numvis x batchsize
%   y                   : numlab x batchsize, y is in binary format
%   objective function  : cross entropy
%

function [w, b] = multiclass_logistic_regression(x, y, l2reg, gradcheck)

if ~exist('gradcheck', 'var'),
    gradcheck = 0;
end

numvis = size(x, 1);
numlab = size(y, 1);

rng('default')

w = 0.001*randn(numvis, numlab);
b = zeros(numlab, 1);
theta = [w(:) ; b(:)];

if gradcheck,
    [diff, ngrad, tgrad] = GradCheck(@(p) multiclass_logistic_regression_sub(p, x, y, l2reg), theta);
end


options.method = 'lbfgs';
options.maxIter = 2000;
options.maxFunEvals = 2000;
options.display = 'off';

opttheta = minFunc(@(p) multiclass_logistic_regression_sub(p, x, y, l2reg), theta, options);
[w, b] = unroll_mlr(opttheta, numvis, numlab);

return;


function [cost, grad] = multiclass_logistic_regression_sub(theta, x, y, l2reg)

numvis = size(x, 1);
numlab = size(y, 1);

batchsize = size(x, 2);
[w, b] = unroll_mlr(theta, numvis, numlab);

% inference
yhat = sigmoid(bsxfun(@plus, w'*x, b));

% objective function
cost = crossentropy(y(:), yhat(:));
% fprintf('cost = %g\n', cost);

% gradient
dw = x*(yhat - y)';
db = sum(yhat - y, 2);

% regularization
cost = cost + 0.5*l2reg*batchsize*sum(w(:).^2);
dw = dw + l2reg*batchsize*w;

grad = [dw(:) ; db(:)];

return;


function [w, b] = unroll_mlr(theta, numvis, numlab)

idx = 0;

w = reshape(theta(idx+1:idx+numvis*numlab), numvis, numlab);
idx = idx + numel(w);

b = theta(idx+1:idx+numlab);
idx = idx + numel(b);

assert(idx == length(theta));

return;


function cost = crossentropy(x, xrec)

xrec = min(xrec, 1-1e-8);
xrec = max(xrec, 1e-8);

cost = -sum(x.*log(xrec)+(1-x).*log(1-xrec));

return;
