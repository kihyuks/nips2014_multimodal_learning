% -------------------------------------------------------------------------
% Classification pipeline for layer-1 text pathway on Flickr
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
%   
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------

function [net, map_val, map_test] = flickr_text_l1(dotest, foldlist, optgpu, doc_length, ...
    numhid, numstep_h, epsilon, sp_reg, sp_target, l2reg, kcd, ...
    maxiter, batchsize, stdinit, upfactor, downfactor)

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
    optgpu = 1;
end
if ~exist('doc_length', 'var'),
    doc_length = 5; % take docs with length > 10
end

% input-output types
typein = 'binary';
typeout = 'binary';

% first layer
if ~exist('numhid', 'var'),
    numhid = 1024;
end
if ~exist('numstep_h', 'var'),
    numstep_h = 1;
end
if ~exist('epsilon', 'var'),
    epsilon = 0.1;
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
    stdinit = 0.005;
end
if ~exist('upfactor', 'var'),
    upfactor = 2;
end
if ~exist('downfactor', 'var'),
    downfactor = 1;
end


dataset = 'flickr_text';
net = cell(1, 1);


% -------------------------------------------------------------------------
%                                     train binary-binary RBM (1st layer)
% -------------------------------------------------------------------------

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

clear weights params history;


% -------------------------------------------------------------------------
%                               test with multi-label logistic classifier
% -------------------------------------------------------------------------

if dotest,
    [xlab, ylab, folds, numdim_img, ~, ~] = load_flickr_lab;
    xlab_text = xlab(numdim_img+1:end, :);
    
    remove_notags = 1;
    if remove_notags,
        % remove examples with no tag
        notagidx = sum(xlab_text, 1) == 0;
        for fold_id = 1:5,
            folds(notagidx, fold_id) = 0;
        end
    end
    
    % inference
    xlab = net{1}.infer(xlab_text);
    
    % run multiclass logistic regression
    [map_val, map_test] = run_mclr(xlab, '', ylab, folds, foldlist, l2reg_list, logdir, sprintf('%s_remove%d', dataset, remove_notags), fname);
else
    map_val = -1;
    map_test = -1;
end

return;
