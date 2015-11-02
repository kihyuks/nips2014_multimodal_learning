function h = rbm_infer_bin_bin(data, weights, params, maxbatchsize)

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
        h_batch = rbm_infer_bin_bin_sub(data_batch, weights, params);
        
        h(:, idx) = single(h_batch);
    end
else
    h = rbm_infer_bin_bin_sub(data, weights, params);
end

return;


function h = rbm_infer_bin_bin_sub(data, weights, params)

batchsize = size(data, 2);

% inference
vishid_bu = params.upfactor*weights.vishid;
hbiasmat = repmat(weights.hidbias, [1 batchsize]);
h = sigmoid(vishid_bu'*data + hbiasmat);

return;