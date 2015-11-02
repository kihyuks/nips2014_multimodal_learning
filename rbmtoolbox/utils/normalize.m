% ===========================================
% normalize samples for real-valued input
% by subtracting mean and dividing by std
% per sample
% ===========================================


function x = normalize(x, eps)

if ~exist('eps', 'var'),
    eps = 1e-3;
end

% subtract mean and divide by std for each example
x = bsxfun(@minus, x, mean(x, 1));
x = bsxfun(@rdivide, x, sqrt(var(x, [], 1) + eps));

return;