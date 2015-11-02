function infer = mdrnn_l2_infer(weights, params, nmf, verbose)

if ~exist('verbose', 'var'),
    verbose = 1;
end

if strcmp(params.typeinx, 'binary') && strcmp(params.typeinz, 'binary'),
    infer = @(x, z)mdrnn_l2_infer_bin_bin(x, z, weights, params, nmf, verbose);
end

return;

