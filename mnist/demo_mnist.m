% common fields
% traintype, numhid, epsilon, eps_decay, sp_type, sp_reg, sp_target, l2reg,
% maxiter, batchsize, stdinit, upfactor_x, upfactor_z, downfactor,
% momentum_change,
%
% training algorithms
% 'pcd'     : ML with CD
%           : usepcd, kcd, negchain
% 'cdpl'    : percloss with CD
%           : kcd
% 'mrnn'    : MP objective with RNN encoding
%           : nmf, nstop, px, pz, alpha (=1)
% 'hybrid'  : pcd + mrnn
%           : usepcd, kcd, nmf, nstop, px, pz, alpha (<1), negchain
%

function demo_mnist(dataset, optgpu, traintype, ...
    numhid, epsilon, eps_decay, sp_reg, sp_target, l2reg, ...
    maxiter, batchsize, stdinit, upfactor_x, upfactor_z, downfactor, ...
    usepcd, kcd, negchain, nmf, nstop, px, pz, alpha)

startup;

if ~exist('dataset', 'var'),
    dataset = 'mnist';
end
if ~exist('optgpu', 'var'),
    optgpu = 1;
end

% input-output type
typeinx = 'binary';
typeinz = 'binary';
typeout = 'binary';

% third layer shared representation
if ~exist('traintype', 'var'),
    traintype = 'pcd';
end
if ~exist('numhid', 'var'),
    numhid = 1024;
end
if ~exist('epsilon', 'var'),
    epsilon = 0.01;
end
if ~exist('eps_decay', 'var'),
    eps_decay = 0.01;
end
if ~exist('sp_reg', 'var'),
    sp_reg = 0.01;
end
if ~exist('sp_target', 'var'),
    sp_target = 0.2;
end
if ~exist('l2reg', 'var'),
    l2reg = 1e-5;
end
if ~exist('maxiter', 'var'),
    maxiter = 300;
end
if ~exist('batchsize', 'var'),
    batchsize = 200;
end
if ~exist('stdinit', 'var'),
    stdinit = 0.01;
end
if ~exist('upfactor_x', 'var'),
    upfactor_x = 1;
end
if ~exist('upfactor_z', 'var'),
    upfactor_z = 1;
end
if ~exist('downfactor', 'var'),
    downfactor = 2;
end
if ~exist('usepcd', 'var'),
    usepcd = 1;
end
if ~exist('kcd', 'var'),
    kcd = 1;
end
if ~exist('negchain', 'var'),
    negchain = 10;
end
if ~exist('nmf', 'var'),
    nmf = 10;
end
if ~exist('nstop', 'var'),
    nstop = 1; % bp by 1 step
end
nstop = min(nstop, nmf);
if ~exist('px', 'var'),
    px = 0;
end
if ~exist('pz', 'var'),
    pz = px;
end
if ~exist('alpha', 'var'),
    alpha = 1;
end


% -------------------------------------------------------------------------
%                   load mnist data and divide into left and right halves
% -------------------------------------------------------------------------

[xtr, ytr, xval, yval, xts, yts, dim] = load_mnist;

% train
xtr = reshape(xtr, [dim, size(xtr, 2)]);
ztr = xtr(:, 15:end, :);
xtr = xtr(:, 1:14, :);
xtr = reshape(xtr, [prod(dim)/2, numel(xtr)/(prod(dim)/2)]);
ztr = reshape(ztr, [prod(dim)/2, numel(ztr)/(prod(dim)/2)]);

% val
xval = reshape(xval, [dim, size(xval, 2)]);
zval = xval(:, 15:end, :);
xval = xval(:, 1:14, :);
xval = reshape(xval, [prod(dim)/2, numel(xval)/(prod(dim)/2)]);
zval = reshape(zval, [prod(dim)/2, numel(zval)/(prod(dim)/2)]);

xts = reshape(xts, [dim, size(xts, 2)]);
zts = xts(:, 15:end, :);
xts = xts(:, 1:14, :);
xts = reshape(xts, [prod(dim)/2, numel(xts)/(prod(dim)/2)]);
zts = reshape(zts, [prod(dim)/2, numel(zts)/(prod(dim)/2)]);


% -------------------------------------------------------------------------
%                          train multimodal RBM for shared representation
% -------------------------------------------------------------------------

% parameter setting
params = struct(...
    'dataset', dataset, ...
    'optgpu', optgpu, ...
    'savedir', savedir, ...
    'traintype', traintype, ...
    'typeinx', typeinx, ...
    'typeinz', typeinz, ...
    'typeout', typeout, ...
    'numvx', size(xtr, 1), ...
    'numvz', size(ztr, 1), ...
    'numhid', numhid, ...
    'eps', epsilon, ...
    'eps_decay', eps_decay, ...
    'sp_type', 'exact', ...
    'sp_reg', sp_reg, ...
    'sp_target', sp_target, ...
    'l2reg', l2reg, ...
    'maxiter', maxiter, ...
    'batchsize', batchsize, ...
    'normalize', false, ...
    'stdinit', stdinit, ...
    'momentum_change', 5, ...
    'momentum_init', 0.33, ...
    'momentum_final', 0.47, ...
    'upfactor_x', upfactor_x, ...
    'upfactor_z', upfactor_z, ...
    'downfactor', downfactor, ...
    'usepcd', usepcd, ...
    'kcd', kcd, ...
    'negchain', negchain, ...
    'nmf', nmf, ...
    'nstop', nstop, ...
    'px', px, ...
    'pz', pz, ...
    'alpha', alpha, ...
    'saveiter', 20, ...
    'verbose', 1);

[fname, params] = mrbm_filename(params);

try
    load(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, maxiter), 'weights', 'params');
    params.fname = fname;
catch
    [weights, params, history] = mrbm_train(xtr, ztr, params);
    save(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, maxiter), 'weights', 'params', 'history');
end

if params.nmf > 0,
    nmf_test = params.nmf;
else
    nmf_test = 10;
end
fname = sprintf('%s_iter_%d', params.fname, maxiter);
infer = @(x, z) mrbm_infer_bin_bin(x, z, weights, params, nmf_test);
cdataset = params.dataset;

clear weights params history;


% -------------------------------------------------------------------------
%                                test with multiclass logistic regression
% -------------------------------------------------------------------------

% inference
htr_mult = infer(xtr, ztr);
hval_mult = infer(xval, zval);
hts_mult = infer(xts, zts);
hts_left = infer(xts, []);
hts_right = infer([], zts);


% multimodal input
htr = htr_mult;
hval = hval_mult;
hts = hts_mult;

[~, acc_val_mult, ~, bestC] = liblinear_wrapper([], htr, ytr, hval, yval, hts, yts);
[~, ~, acc_ts_mult] = liblinear_wrapper(bestC, [htr hval], [ytr(:) ; yval(:)], hval, yval, hts, yts);

% unimodal input: left
htr = htr_mult;
hval = hval_mult;
hts = hts_left;

[~, ~, acc_ts_left] = liblinear_wrapper(bestC, [htr hval], [ytr(:) ; yval(:)], hval, yval, hts, yts);

% unimodal input: right
htr = htr_mult;
hval = hval_mult;
hts = hts_right;

[~, ~, acc_ts_right] = liblinear_wrapper(bestC, [htr hval], [ytr(:) ; yval(:)], hval, yval, hts, yts);

% log
fprintf('err: val = %g, test = %g, test (left) = %g, test (right) = %g (C=%g, %s)\n', ...
    100-acc_val_mult, 100-acc_ts_mult, 100-acc_ts_left, 100-acc_ts_right, bestC, fname);
fid = fopen(sprintf('%s/%s.txt', logdir, cdataset), 'a+');
fprintf(fid, 'err: val = %g, test = %g, test (left) = %g, test (right) = %g (C=%g, %s)\n', ...
    100-acc_val_mult, 100-acc_ts_mult, 100-acc_ts_left, 100-acc_ts_right, bestC, fname);
fclose(fid);


return;


