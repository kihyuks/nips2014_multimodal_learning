function h = rbm_infer_real_bin(data, weights, params, maxbatchsize)

if ~isfield(params, 'upfactor'),
    params.upfactor = 1;
end
if ~exist('maxbatchsize', 'var'),
    maxbatchsize = 10000;
end

batchsize = size(data, 2);

if batchsize > maxbatchsize,
    h = single(zeros(params.numhid, batchsize));
    
    for i = 1:ceil(batchsize/maxbatchsize),
        idx = (i-1)*maxbatchsize+1:min(batchsize, i*maxbatchsize);
        data_batch = data(:, idx);
        h_batch = rbm_infer_real_bin_sub(data_batch, weights, params);
        
        h(:, idx) = single(h_batch);
    end
else
    h = rbm_infer_real_bin_sub(data, weights, params);
end

return;


function h = rbm_infer_real_bin_sub(data, weights, params)

% normailze
if params.normalize,
    if ~isfield(params, 'epsnorm'),
        params.epsnorm = 1e-3;
    end
    data = normalize(data, params.epsnorm);
end

% inference
vishid_bu = params.upfactor*bsxfun(@rdivide, weights.vishid, weights.stds);
hbiasmat = repmat(weights.hidbias, [1 batchsize]);
h = sigmoid(vishid_bu'*data + hbiasmat);

return;