function diff = gradcheck_mrnn(nmf)

addpath ~/libdeepnets2/common/utils/
if ~exist('nmf', 'var'),
    nmf = 0;
end
params.numvx = 10;
params.numvz = 12;
params.numhid = 8;
params.nmf = nmf;
params.upfactor_x = 1;
params.upfactor_z = 1;
params.downfactor = 2;
params.sp_target = 0.1;
params.sp_reg = 0.1;
params.px = 0.2;
params.pz = 0.2;
params.sp_type = 'exact';

batchsize = 100;

data_x = rand(params.numvx, batchsize);
data_z = rand(params.numvz, batchsize);

weights = struct;
weights.vxhid = 0.1*randn(params.numvx, params.numhid);
weights.vzhid = 0.1*randn(params.numvz, params.numhid);
weights.hidbias = zeros(params.numhid, 1);
weights.vxbias = zeros(params.numvx, 1);
weights.vzbias = zeros(params.numvz, 1);

theta = roll_rbm_multimodal(weights);


[cost, pos] = cost_mrnn(theta, data_x, data_z, params);

addpath ~/scratch/kihyuks/library/GradCheck/;

[diff, ngrad, tgrad] = GradCheck(@(p) cost_mrnn(p, data_x, data_z, params), theta);

ngrad = unroll_rbm_multimodal(ngrad, params);
tgrad = unroll_rbm_multimodal(tgrad, params);

fnames = fieldnames(ngrad);
for i = 1:length(fnames),
    nA = getfield(ngrad, fnames{i});
    tA = getfield(tgrad, fnames{i});
    fprintf('%s diff = %g\n', fnames{i}, norm(nA(:) - tA(:))/norm(nA(:) + tA(:)));
end


return;


function [cost, pos] = cost_mrnn(theta, data_x, data_z, params)

rng('default');

nmf = params.nmf;
batchsize = size(data_x, 2);
orig = 1;

weights = unroll_rbm_multimodal(theta, params);

% replicate for gradient
pos = replicate_struct(weights, 0);

vxhid_bu = params.upfactor_x*weights.vxhid;
vzhid_bu = params.upfactor_z*weights.vzhid;
vzhid_td = params.downfactor*weights.vzhid;

hbiasmat = repmat(weights.hidbias, [1 batchsize]);
zbiasmat = repmat(weights.vzbias, [1 batchsize]);

% corrupt input data
mask_z = sample_bernoulli(params.pz*ones(size(data_z)), 0);
data_z_corrupt = data_z.*mask_z;

hidprob = zeros(params.numhid, batchsize, nmf+1);
vzprob = zeros(params.numvz, batchsize, nmf+1);

wx = vxhid_bu'*data_x;

% hidden
if orig,
    upfactor_x_init = (params.upfactor_x + params.upfactor_z*(1-params.pz))/params.upfactor_x;
    upfactor_z_init = (params.upfactor_z*params.pz)/params.upfactor_z;
    hidprob(:, :, 1) = sigmoid(upfactor_x_init*wx + upfactor_z_init*(vzhid_bu'*data_z_corrupt) + hbiasmat);
else
    hidprob(:, :, 1) = sigmoid(params.upfactor_x*wx + params.upfactor_z*(vzhid_bu'*(data_z_corrupt)) + hbiasmat);
end
for i = 1:nmf,
    % visible (z)
    vzprob(:, :, i) = sigmoid(vzhid_td*hidprob(:, :, i) + zbiasmat);
    vzprob(:, :, i) = vzprob(:, :, i).*(1-mask_z) + data_z_corrupt;
    
    % hidden
    hidprob(:, :, i+1) = sigmoid(wx + vzhid_bu'*vzprob(:, :, i) + hbiasmat);
end
% visible (z)
vzprob(:, :, nmf+1) = sigmoid(vzhid_td*hidprob(:, :, nmf+1) + zbiasmat);
vzprob(:, :, nmf+1) = vzprob(:, :, nmf+1).*(1-mask_z) + data_z_corrupt;

% reconstruction
cost = cross_entropy(data_z, vzprob(:, :, nmf+1))/batchsize;

% compute gradient (full backprop)
dobj = (data_z - vzprob(:, :, nmf+1))/batchsize;

for i = nmf+1:-1:1,
    % vz -> hid
    pos.vzhid = pos.vzhid + params.downfactor*dobj*hidprob(:, :, i)';
    pos.vzbias = pos.vzbias + sum(dobj, 2);
    hmh = hidprob(:, :, i).*(1-hidprob(:, :, i));
    dobj = hmh.*(vzhid_td'*dobj);
    
    if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg > 0,
        mh = sum(hidprob(:, :, i),2)/batchsize;
        mhtmp = params.sp_target./mh - (1-params.sp_target)./(1-mh);        
        cost = cost + params.sp_reg*sum((params.sp_target*log(params.sp_target./mh) + (1-params.sp_target)*log((1-params.sp_target)./(1-mh))));
        dobj = dobj + params.sp_reg*bsxfun(@times, mhtmp, hmh)/batchsize;
    end
    
    if i == 1,
        if orig,
            % hid -> vx (upfactor), bias
            pos.vxhid = pos.vxhid + params.upfactor_x*upfactor_x_init*data_x*dobj';
            pos.vzhid = pos.vzhid + params.upfactor_z*upfactor_z_init*data_z_corrupt*dobj';
            pos.hidbias = pos.hidbias + sum(dobj, 2);
        else
            % hid -> vx (upfactor), bias
            pos.vxhid = pos.vxhid + params.upfactor_x*data_x*dobj';
            pos.vzhid = pos.vzhid + params.upfactor_z*(data_z_corrupt)*dobj';
            pos.hidbias = pos.hidbias + sum(dobj, 2);
        end
    else
        % hid -> vx, bias
        pos.vxhid = pos.vxhid + params.upfactor_x*data_x*dobj';        
        pos.vzhid = pos.vzhid + params.upfactor_z*vzprob(:, :, i-1)*dobj';
        pos.hidbias = pos.hidbias + sum(dobj, 2);
        dobj = (1-mask_z).*vzprob(:, :, i-1).*(1-vzprob(:, :, i-1)).*(vzhid_bu*dobj);
    end
end


pos = replicate_struct(pos, -1);
pos = roll_rbm_multimodal(pos);

return;


function weights = unroll_rbm_multimodal(theta, params)

idx = 0;
weights = struct;
numvx = params.numvx;
numvz = params.numvz;
numhid = params.numhid;

weights.vxhid = reshape(theta(idx+1:idx+numvx*numhid), numvx, numhid);
idx = idx + numel(weights.vxhid);

weights.vzhid = reshape(theta(idx+1:idx+numvz*numhid), numvz, numhid);
idx = idx + numel(weights.vzhid);

weights.hidbias = theta(idx+1:idx+numhid);
idx = idx + numel(weights.hidbias);

weights.vxbias = theta(idx+1:idx+numvx);
idx = idx + numel(weights.vxbias);

weights.vzbias = theta(idx+1:idx+numvz);
idx = idx + numel(weights.vzbias);

assert(idx == length(theta));


return;


function theta = roll_rbm_multimodal(weights)

theta = [];
theta = [theta ; weights.vxhid(:)];
theta = [theta ; weights.vzhid(:)];
theta = [theta ; weights.hidbias(:)];
theta = [theta ; weights.vxbias(:)];
theta = [theta ; weights.vzbias(:)];

return;