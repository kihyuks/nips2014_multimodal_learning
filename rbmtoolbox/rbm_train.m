function [weights, params, history, infer, grad] = rbm_train(xtrain, params)


% fill in missing values
params = fillin_params(params);

% input-output types
typein = params.typein;
typeout = params.typeout;

% filename to save
if ~isfield(params,'fname'),
    params.fname = sprintf('%s_%s_%s_v%d_h%d_step_%d_eps%g_l2reg%g_%s_target%g_reg%g_pcd%d_kcd%d_bs%d', ...
        params.dataset, params.typein, params.typeout, params.numvis, ...
        params.numhid, params.numstep_h, params.eps, ...
        params.l2reg, params.sp_type, params.sp_target, params.sp_reg, ...
        params.usepcd, params.kcd, params.batchsize);
end
params.fname_save = sprintf('%s_iter_%d_date_%s', params.fname, params.maxiter, datestr(now, 30));


% learning start
rng('default');

if strcmp(typein, 'real') && strcmp(typeout, 'binary'),
    % real-binary RBM
    [weights, params, grad, history] = rbm_train_real_bin(xtrain, params);
    infer = @(x)rbm_infer_real_bin(x, weights, params);
    
elseif strcmp(typein, 'real') && strcmp(typeout, 'step'),
    % real-stepped sigmoid RBM
    [weights, params, grad, history] = rbm_train_real_step(xtrain, params);
    infer = @(x)rbm_infer_real_step(x, weights, params);
    
elseif strcmp(typein, 'binary') && strcmp(typeout, 'binary'),
    % binary-binary RBM
    [weights, params, grad, history] = rbm_train_bin_bin(xtrain, params);
    infer = @(x)rbm_infer_bin_bin(x, weights);
    
elseif strcmp(typein, 'binary') && strcmp(typeout, 'step'),
    % binary-stepped sigmoid RBM
    [weights, params, grad, history] = rbm_train_bin_step(xtrain, params);
    infer = @(x)rbm_infer_bin_step(x, weights, params);
    
else
    error('undefined RBM types');
end


weights = gpu2cpu_struct(weights);
history = gpu2cpu_struct(history);
grad = gpu2cpu_struct(grad);

return;
