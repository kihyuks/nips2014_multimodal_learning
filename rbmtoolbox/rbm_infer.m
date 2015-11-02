function [params, infer] = rbm_infer(weights, params)

% input-output types
typein = params.typein;
typeout = params.typeout;


% learning start
if strcmp(typein, 'real') && strcmp(typeout, 'binary'),
    % real-binary RBM
    infer = @(x)rbm_infer_real_bin(x, weights, params);
    
elseif strcmp(typein, 'real') && strcmp(typeout, 'step'),
    % real-stepped sigmoid RBM
    infer = @(x)rbm_infer_real_step(x, weights, params);
    
elseif strcmp(typein, 'binary') && strcmp(typeout, 'binary'),
    % binary-binary RBM
    infer = @(x)rbm_infer_bin_bin(x, weights, params);
    
elseif strcmp(typein, 'binary') && strcmp(typeout, 'step'),
    % binary-stepped sigmoid RBM
    infer = @(x)rbm_infer_bin_step(x, weights, params);
    
else
    error('undefined RBM types');
end

params.infer = infer;

return;
