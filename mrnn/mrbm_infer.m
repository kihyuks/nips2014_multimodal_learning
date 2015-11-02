function infer = mrbm_infer(weights, params, nmf)

if strcmp(params.typeinx, 'binary') && strcmp(params.typeinz, 'binary'),
    infer = @(x, z)mrbm_infer_bin_bin(x, z, weights, params, nmf);
end

return;
