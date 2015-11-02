function [weights, grad] = update_params(weights, grad, pos, neg, momentum, epsilon, usepcd)

fname = fieldnames(weights);

for i = 1:length(fname),
    % load fields
    pA = getfield(pos, fname{i});
    nA = getfield(neg, fname{i});
    gA = getfield(grad, fname{i});
    A = getfield(weights, fname{i});
    
    if usepcd,
        gA = momentum*gA + (1-momentum)*epsilon*pA;
        A = A + (gA - epsilon*nA);
    else
        gA = momentum*gA + (1-momentum)*epsilon*(pA - nA);
        A = A + gA;
    end
    
    % update accumulate parameter and weights
    grad = setfield(grad, fname{i}, gA);
    weights = setfield(weights, fname{i}, A);
end

return;
