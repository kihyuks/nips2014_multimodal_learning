% -------------------------------------------------------------------------
% multimodal RBM with two input modalities (both binary)
% traintype =
%   'pcd', 'cdpl', 'mrnn', 'hybrid'
% -------------------------------------------------------------------------


function [weights, params, history, grad] = mrbm_train(xtr, ztr, params)

% filename to save
params.fname_save = sprintf('%s_iter_%d_date_%s', params.fname, params.maxiter, datestr(now, 30));


% learning start
rng('default');

switch params.traintype,
    case 'pcd',
        % multimodal RBM with ML using PCD
        [weights, params, grad, history] = mrbm_train_bin_bin(xtr, ztr, params);
    case 'cdpl',
        % multimodal RBM with minVI using CD-percLoss
        [weights, params, grad, history] = mrbm_train_bin_bin_cdpl(xtr, ztr, params);
    case {'mrnn', 'hybrid'},
        % multimodal RBM with minVI using multi-prediction training
        [weights, params, grad, history] = mrnn_train_bin_bin(xtr, ztr, params);
    otherwise
        error('undefined model types!');
end


weights = gpu2cpu_struct(weights);
history = gpu2cpu_struct(history);
grad = gpu2cpu_struct(grad);


return;
