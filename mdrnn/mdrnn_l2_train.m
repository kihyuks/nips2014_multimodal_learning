% -------------------------------------------------------------------------
% multimodal RNN with two input modalities
% -------------------------------------------------------------------------

function [weights, params, history] = mdrnn_l2_train(xtr, ztr, weights, params)

% input-output types
traintype = params.traintype;
typeinx = params.typeinx;
typeinz = params.typeinz;

% filename to save
params.fname_save = sprintf('%s_iter_%d_date_%s', params.fname, params.maxiter, datestr(now, 30));


% learning start
switch traintype,
    case 'mdrnn',
        if strcmp(typeinx, 'binary') && strcmp(typeinz, 'binary'),
            [weights, params, grad, history] = mdrnn_l2_train_bin_bin(xtr, ztr, weights, params);
        end
    otherwise
        error('undefined model');
end


weights = gpu2cpu_struct(weights);
history = gpu2cpu_struct(history);
grad = gpu2cpu_struct(grad);


return;
