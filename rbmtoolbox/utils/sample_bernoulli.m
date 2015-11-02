% -------------------------------------------------------------------------
% draw sample from bernoulli distribution x
% -------------------------------------------------------------------------

function y = sample_bernoulli(x, optgpu)

if ~exist('optgpu','var'),
    optgpu = 0;
end

y = x > rand(size(x));
if optgpu,
    y = gpuArray(single(y));
else
    y = double(y);
end

return;

