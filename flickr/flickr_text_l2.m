% -------------------------------------------------------------------------
% Classification pipeline for layer-2 text pathway on Flickr
%   using binary-binary RBM
%
%   dotest (1/0), foldlist (1:5), optgpu (1/0),
%   doc_length  : minimum length for tags (5)
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
%       (same applies to param"2")
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------

function [net, map_val, map_test] = flickr_text_l2(dotest, foldlist, optgpu, doc_length, ...
    numhid, numstep_h, epsilon, sp_reg, sp_target, l2reg, kcd, maxiter, batchsize, stdinit, upfactor, downfactor, ...
    numhid2, numstep_h2, epsilon2, sp_reg2, sp_target2, l2reg2, kcd2, maxiter2, batchsize2, stdinit2, upfactor2, downfactor2)

startup;
l2reg_list = [1e-5 1e-4 1e-3 1e-2 1e-1]; % cross validation


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
    optgpu = 5;
end
if ~exist('doc_length', 'var'),
    doc_length = 5; % take docs with length >= doc_length
end

% input-output types
typein = 'binary';
typeout = 'binary'; % binary hidden

typein2 = 'binary';
typeout2 = 'binary'; % binary hidden


% first layer
if ~exist('numhid', 'var'),
    numhid = 512;
end
if ~exist('numstep_h', 'var'),
    numstep_h = 5; % > 1 for stepped-sigmoid hidden
end
if ~exist('epsilon', 'var'),
    epsilon = 0.5;
end
if ~exist('sp_reg', 'var'),
    sp_reg = 0.01;
end
if ~exist('sp_target', 'var'),
    sp_target = 0.2;
end
if ~exist('l2reg', 'var'),
    l2reg = 0;
end
if ~exist('kcd', 'var'),
    kcd = 1;
end
if ~exist('maxiter', 'var'),
    maxiter = 200;
end
if ~exist('batchsize', 'var'),
    batchsize = 100;
end
if ~exist('stdinit', 'var'),
    stdinit = 0.01;
end
if ~exist('upfactor', 'var'),
    upfactor = 2;
end
if ~exist('downfactor', 'var'),
    downfactor = 1;
end

% second layer
if ~exist('numhid2', 'var'),
    numhid2 = 512;
end
if ~exist('numstep_h2', 'var'),
    numstep_h2 = 5; % > 1 for stepped-sigmoid hidden
end
if ~exist('epsilon2', 'var'),
    epsilon2 = 0.5;
end
if ~exist('sp_reg2', 'var'),
    sp_reg2 = 0.01;
end
if ~exist('sp_target2', 'var'),
    sp_target2 = 0.2;
end
if ~exist('l2reg2', 'var'),
    l2reg2 = 1e-5;
end
if ~exist('kcd2', 'var'),
    kcd2 = 1;
end
if ~exist('maxiter2', 'var'),
    maxiter2 = 100;
end
if ~exist('batchsize2', 'var'),
    batchsize2 = 100;
end
if ~exist('stdinit2', 'var'),
    stdinit2 = 0.01;
end
if ~exist('upfactor2', 'var'),
    upfactor2 = 2;
end
if ~exist('downfactor2', 'var'),
    downfactor2 = 1;
end


dataset = 'flickr_text';
net = cell(2, 1);


% -------------------------------------------------------------------------
%                                     train binary-binary RBM (1st layer)
% -------------------------------------------------------------------------

if ~exist('params1', 'var') || isempty(params1),
    params = struct(...
        'dataset', dataset, ...
        'savedir', savedir, ...
        'optgpu', optgpu, ...
        'typein', typein, ...
        'typeout', typeout, ...
        'doc_length', doc_length, ...
        'numvis', 2000, ...
        'numhid', numhid, ...
        'numstep_h', numstep_h, ...
        'eps', epsilon, ...
        'eps_decay', 0.01, ...
        'sp_type', 'exact', ...
        'sp_reg', sp_reg, ...
        'sp_target', sp_target, ...
        'l2reg', l2reg, ...
        'usepcd', 1, ...
        'kcd', kcd, ...
        'negchain', 500, ...
        'maxiter', maxiter, ...
        'saveiter', maxiter, ...
        'batchsize', batchsize, ...
        'normalize', false, ...
        'stdinit', stdinit, ...
        'momentum_change', 100, ...
        'momentum_init', 0.33, ...
        'momentum_final', 0.5, ...
        'upfactor', upfactor, ...
        'downfactor', downfactor);
else
    params = params1;
end

params = fillin_params(params);

params.fname = sprintf('%s_%s_%s_v_%d_h_%d_step_%d_ln_%d_eps_%g_l2r_%g_%s_target_%g_reg_%g_pcd_%d_kcd_%d_bs_%d_init_%g_up_%d_down_%d', ...
    params.dataset, params.typein, params.typeout, params.numvis, params.numhid, ...
    params.numstep_h, params.doc_length, params.eps, params.l2reg, params.sp_type, ...
    params.sp_target, params.sp_reg, params.usepcd, params.kcd, params.batchsize, ...
    params.stdinit, params.upfactor, params.downfactor);

try
    load(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, params.maxiter), 'weights', 'params');
    fprintf('load first layer text dictionary\n');
catch
    % load data
    xunlab = load_flickr_unlab_text;
    xunlab(:, sum(xunlab, 1) < doc_length) = []; % remove examples with no tags
    xunlab = single(xunlab);
    
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

savedir = sprintf('%s/%s', savedir, fname);
if ~exist(savedir, 'dir'),
    mkdir(savedir);
end

clear weights params history;


% -------------------------------------------------------------------------
%                                     train binary-binary RBM (2nd layer)
% -------------------------------------------------------------------------

if ~exist('params2', 'var') || isempty(params2),
    params = struct(...
        'dataset', sprintf('%s_l2', dataset), ...
        'optgpu', optgpu, ...
        'savedir', savedir, ...
        'typein', typein2, ...
        'typeout', typeout2, ...
        'numvis', net{1}.params.numhid, ...
        'numhid', numhid2, ...
        'numstep_h', numstep_h2, ...
        'eps', epsilon2, ...
        'eps_decay', 0.01, ...
        'sp_type', 'exact', ...
        'sp_reg', sp_reg2, ...
        'sp_target', sp_target2, ...
        'l2reg', l2reg2, ...
        'usepcd', 1, ...
        'negchain', 500, ...
        'kcd', kcd2, ...
        'maxiter', maxiter2, ...
        'saveiter', maxiter2, ...
        'batchsize', batchsize2, ...
        'normalize', false, ...
        'stdinit', stdinit2, ...
        'std_learn', 0, ...
        'draw_sample', 1, ...
        'momentum_change', 5, ...
        'momentum_init', 0.33, ...
        'momentum_final', 0.5, ...
        'upfactor', upfactor2, ...
        'downfactor', downfactor2);
else
    params = params2;
end

params = fillin_params(params);

params.fname = sprintf('%s_%s_%s_v_%d_h_%d_step_%d_%d_eps_%g_l2r_%g_%s_target_%g_reg_%g_pcd_%d_kcd_%d_bs_%d_init_%g_draw_%d_up_%d_down_%d', ...
    params.dataset, params.typein, params.typeout, params.numvis, params.numhid, ...
    params.numstep_v, params.numstep_h, params.eps, params.l2reg, params.sp_type, ...
    params.sp_target, params.sp_reg, params.usepcd, params.kcd, params.batchsize, ...
    params.stdinit, params.draw_sample, params.upfactor, params.downfactor);

try
    load(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, params.maxiter), 'weights', 'params');
    fprintf('load second layer text dictionary\n');
catch
    % load data
    xunlab = load_flickr_unlab_text;
    xunlab(:, sum(xunlab, 1) < doc_length) = []; % remove examples with no tags
    xunlab = single(xunlab);
    
    % inference
    xunlab = net{1}.infer(xunlab);
    xunlab = single(xunlab);
    
    % rbm training
    [weights, params, history] = rbm_train(xunlab, params);
    save(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, params.maxiter), 'weights', 'params', 'history');
    
    clear xunlab;
end

fname = sprintf('%s_iter_%d', params.fname, params.maxiter);
[~, infer] = rbm_infer(weights, params);

net{2}.weights = weights;
net{2}.params = params;
net{2}.infer = infer;

clear weights params history;


% -------------------------------------------------------------------------
%                               test with multi-label logistic classifier
% -------------------------------------------------------------------------

if dotest,
    [xlab, ylab, folds, numdim_img, ~, ~] = load_flickr_lab;
    xlab = xlab(numdim_img+1:end, :);
    
    remove_notags = 1;
    if remove_notags,
        % remove examples with no tag
        notagidx = sum(xlab, 1) == 0;
        for fold_id = 1:5,
            folds(notagidx, fold_id) = 0;
        end
    end
    
    % inference
    xlab = net{1}.infer(xlab);
    xlab = net{2}.infer(xlab);
    
    % run multiclass logistic regression
    [map_val, map_test] = run_mclr(xlab, '', ylab, folds, foldlist, l2reg_list, logdir, sprintf('%s_remove%d', dataset, remove_notags), fname);
end

return;
