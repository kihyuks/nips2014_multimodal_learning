% -------------------------------------------------------------------------
% Classification pipeline for layer-1 image pathway on Flickr
%   using gaussian-stepped sigmoid RBM
%
%   dotest (1/0), foldlist (1:5), optgpu (1/0),
%   numhid      : number of latent variables
%   numstep_h   : number of stepped sigmoid units, >= 1
%   epsilon     : learning rate
%   sp_reg      : sparsity regularization weight
%   sp_target   : sparsity target, >= 0, <= 1
%   l2reg       : l2 weight decay weight
%   kcd         : number of CD steps, >= 1
%   maxiter     : max epoch
%   batchsize   : mini-batch size
%   stdinit     : initialization weight (std of gaussian)
%   upfactor    : multiplier for hidden unit inference
%   downfactor  : multiplier for visible unit inference
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------

function [net, map_val, map_test] = flickr_img_l1(dotest, foldlist, optgpu, ...
    numhid, numstep_h, epsilon, sp_reg, sp_target, l2reg, kcd, ...
    maxiter, batchsize, stdinit, upfactor, downfactor)

startup;
l2reg_list = [1e-3 3e-3 1e-2 3e-2 1e-1]; % cross validation


% -------------------------------------------------------------------------
%                                                    initialize variables
% -------------------------------------------------------------------------

if ~exist('dotest', 'var'),
    dotest = 1;
end
if ~exist('foldlist', 'var') || isempty(foldlist),
    foldlist = 1:5;
end
if ~exist('optgpu', 'var'),
    optgpu = 1;
end

% input-output types
typein = 'real';    % real-valued
typeout = 'step';   % stepped sigmoid

% first layer
if ~exist('numhid', 'var'),
    numhid = 1024;
end
if ~exist('numstep_h', 'var'),
    numstep_h = 5;
end
if ~exist('epsilon', 'var'),
    epsilon = 0.001;
end
if ~exist('sp_reg', 'var'),
    sp_reg = 1;
end
if ~exist('sp_target', 'var'),
    sp_target = 0.2;
end
if ~exist('l2reg', 'var'),
    l2reg = 1e-5;
end
if ~exist('kcd', 'var'),
    kcd = 1;
end
if ~exist('maxiter', 'var'),
    maxiter = 300;
end
if ~exist('batchsize', 'var'),
    batchsize = 100;
end
if ~exist('stdinit', 'var'),
    stdinit = 0.005;
end
if ~exist('upfactor', 'var'),
    upfactor = 2;
end
if ~exist('downfactor', 'var'),
    downfactor = 1;
end


dataset = 'flickr_img';
net = cell(1, 1);


% -------------------------------------------------------------------------
%                          train gaussian-stepped sigmoid RBM (1st layer)
% -------------------------------------------------------------------------

params = struct(...
    'dataset', dataset, ...
    'optgpu', optgpu, ...
    'savedir', savedir, ...
    'typein', typein, ...
    'typeout', typeout, ...
    'numvis', 3857, ...
    'numhid', numhid, ...
    'numstep_h', numstep_h, ...
    'eps', epsilon, ...
    'eps_decay', 0.01, ...
    'sp_type', 'approx', ...
    'sp_reg', sp_reg, ...
    'sp_target', sp_target, ...
    'l2reg', l2reg, ...
    'usepcd', 0, ...
    'kcd', kcd, ...
    'maxiter', maxiter, ...
    'saveiter', maxiter, ...
    'batchsize', batchsize, ...
    'normalize', false, ...
    'stdinit', stdinit, ...
    'std_learn', 0, ...
    'momentum_change', 5, ...
    'momentum_init', 0.33, ...
    'momentum_final', 0.5, ...
    'upfactor', upfactor, ...
    'downfactor', downfactor);

params = fillin_params(params);

params.fname = sprintf('%s_%s_%s_v_%d_h_%d_step_%d_eps_%g_l2r_%g_%s_target_%g_reg_%g_pcd_%d_kcd_%d_bs_%d_init_%g_up_%d_down_%d', ...
    params.dataset, params.typein, params.typeout, params.numvis, params.numhid, ...
    params.numstep_h, params.eps, params.l2reg, params.sp_type, params.sp_target, params.sp_reg, ...
    params.usepcd, params.kcd, params.batchsize, params.stdinit, params.upfactor, params.downfactor);


% load mean and std for global preprocessing
[m_global, stds_global] = compute_mean_std(0, 10000);
try
    load(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, params.maxiter), 'weights', 'params');
    fprintf('load first layer image dictionary\n');
catch
    % load data
    [xunlab, numdim_img] = load_flickr_unlab(0, 2000);
    xunlab = xunlab(1:numdim_img, :);
    
    % preprocessing
    xunlab = bsxfun(@rdivide, bsxfun(@minus, xunlab, m_global), stds_global);
    
    % rbm training
    [weights, params, history] = rbm_train(xunlab, params);
    save(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, params.maxiter), 'weights', 'params', 'history');
    
    clear xunlab;
end

fname = sprintf('%s_iter_%d', params.fname, params.maxiter);
[~, infer] = rbm_infer(weights, params);

net{1}.weights = weights;
net{1}.params = params;
net{1}.infer = infer;

clear weights params history;


% -------------------------------------------------------------------------
%                               test with multi-label logistic classifier
% -------------------------------------------------------------------------

if dotest,
    [xlab, ylab, folds, numdim_img, ~, ~] = load_flickr_lab;
    xlab_img = xlab(1:numdim_img, :);
    
    % preprocessing & inference
    xlab_img = bsxfun(@rdivide, bsxfun(@minus, xlab_img, m_global), stds_global);
    xlab = net{1}.infer(xlab_img);
    
    % run multiclass logistic regression
    [map_val, map_test, l2reg] = run_mclr(xlab, '', ylab, folds, foldlist, l2reg_list, logdir, dataset, fname);
else
    map_val = -1;
    map_test = -1;
end

return;
