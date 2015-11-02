function infer = mdrnn_l3_infer(weights, params, nmf, verbose)

if ~exist('verbose', 'var'),
    verbose = 1;
end

if strcmp(params.typeinx, 'real') && strcmp(params.typeinz, 'binary'),
    infer = @(x, z)mdrnn_l3_infer_real_bin(x, z, weights, params, nmf, verbose);
end

return;
