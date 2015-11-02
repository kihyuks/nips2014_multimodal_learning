function [hfirst, hprob, hexp] = rbm_infer_real_step(data, weights, params, maxbatchsize)

if ~isfield(params, 'upfactor'),
    params.upfactor = 1;
end
if ~exist('maxbatchsize', 'var'),
    maxbatchsize = 10000;
end

batchsize = size(data, 2);

if batchsize > maxbatchsize,
    hprob = single(zeros(params.numhid, batchsize));
    hexp = single(zeros(params.numhid, batchsize));
    hfirst = single(zeros(params.numhid, batchsize));
    
    for i = 1:ceil(batchsize/maxbatchsize),
        idx = (i-1)*maxbatchsize+1:min(batchsize, i*maxbatchsize);
        data_batch = data(:, idx);
        [hprob_batch, hexp_batch, hfirst_batch] = rbm_infer_real_step_sub(data_batch, weights, params);
        
        hprob(:, idx) = single(hprob_batch);
        hexp(:, idx) = single(hexp_batch);
        hfirst(:, idx) = single(hfirst_batch);
    end
else
    [hprob, hexp, hfirst] = rbm_infer_real_step_sub(data, weights, params);
end

return;


function [hprob, hexp, hfirst] = rbm_infer_real_step_sub(data, weights, params)

batchsize = size(data, 2);

% normailze
if params.normalize,
    if ~isfield(params, 'epsnorm'),
        params.epsnorm = 1e-3;
    end
    data = normalize(data, params.epsnorm);
end

% inference
hidbias_offset = - (1:params.numstep_h) + 0.5;
hidbias_offset = reshape(hidbias_offset, [1, 1, length(hidbias_offset)]);

vishid_bu = params.upfactor*bsxfun(@rdivide, weights.vishid, weights.stds);
hbiasmat = repmat(weights.hidbias, [1 batchsize]);
hexp = vishid_bu'*data + hbiasmat;
hprob = sigmoid(bsxfun(@plus, hexp, hidbias_offset));
hprob = sum(hprob, 3);
hfirst = sigmoid(hexp - 0.5);

return;