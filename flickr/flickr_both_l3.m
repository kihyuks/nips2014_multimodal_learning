% -------------------------------------------------------------------------
% Classification pipeline for multimodal RNN on Flickr
%   a-1.    image pathway layer1, layer2
%   a-2.    text pathway layer1, layer2
%   b.      pretrain top shared layer
%   c.      finetuning top 2 layers (image layer2, text layer2, top)
%   d.      finetuning whole network
%
%   dotest (1/0), foldlist (1:5), optgpu (1/0), dojoint & dofinetune (1/0)
%   numhid      : number of latent variables
%   epsilon     : learning rate
%   eps_decay   : learning rate decay
%   sp_reg      : sparsity regularization weight
%   sp_target   : sparsity target, >= 0, <= 1
%   l2reg       : l2 weight decay weight
%   nmf         : number of mean-field steps, >= 1
%   maxiter     : max epoch
%   batchsize   : mini-batch size
%   px, pz      : droprate for multi-prediction training
%       (sampe applies to param"_jt" and param"_ft"
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------

function [map_val_mult, map_test_mult, map_val_uni, map_test_uni] = flickr_both_l3(dotest, foldlist, optgpu, ...
    numhid, epsilon, eps_decay, sp_reg, sp_target, l2reg, nmf, maxiter, batchsize, px, pz, ...
    dojoint, epsilon_jt, eps_decay_jt, sp_reg_jt, l2reg_jt, nmf_jt, maxiter_jt, px_jt, pz_jt, ...
    dofinetune, epsilon_ft, eps_decay_ft, sp_reg_ft, l2reg_ft, nmf_ft, maxiter_ft, px_ft, pz_ft)

startup;
l2reg_test = 0.001;
evalmode = 'test';

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
doc_length = 2;
imgperbatch = 10000;


% input-output type
typeinx = 'binary';
typeinz = 'binary';
typeout = 'binary';
traintype = 'mrnn';
traintype_jt = 'mdrnn';
traintype_ft = 'mdrnn_xtoz';


% pretraining
if ~exist('numhid', 'var'),
    numhid = 2048;
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
if ~exist('nmf', 'var'),
    nmf = 10;
end
if ~exist('maxiter', 'var'),
    maxiter = 200;
end
if ~exist('batchsize', 'var'),
    batchsize = 200;
end
if ~exist('px', 'var'),
    px = 0;
end
if ~exist('pz', 'var'),
    pz = 0;
end


% joint training
if ~exist('dojoint', 'var'),
    dojoint = 0;
end
if ~exist('epsilon_jt', 'var'),
    epsilon_jt = 0.01;
end
if ~exist('eps_decay_jt', 'var'),
    eps_decay_jt = 0.01;
end
if ~exist('sp_reg_jt', 'var'),
    sp_reg_jt = 0.01;
end
if ~exist('l2reg_jt', 'var'),
    l2reg_jt = 1e-5;
end
if ~exist('nmf_jt', 'var'),
    nmf_jt = nmf;
end
if ~exist('maxiter_jt', 'var'),
    maxiter_jt = 200;
end
if ~exist('px_jt', 'var'),
    px_jt = 0;
end
if ~exist('pz_jt', 'var'),
    pz_jt = 0;
end

% fine-tuning
if ~exist('dofinetune', 'var'),
    dofinetune = 0;
end
if ~exist('epsilon_ft', 'var'),
    epsilon_ft = 0.03;
end
if ~exist('eps_decay_ft', 'var'),
    eps_decay_ft = 0.01;
end
if ~exist('sp_reg_ft', 'var'),
    sp_reg_ft = 0.01;
end
if ~exist('l2reg_ft', 'var'),
    l2reg_ft = 1e-5;
end
if ~exist('nmf_ft', 'var'),
    nmf_ft = nmf;
end
if ~exist('maxiter_ft', 'var'),
    maxiter_ft = 200;
end
if ~exist('px_ft', 'var'),
    px_ft = 1;
end
if ~exist('pz_ft', 'var'),
    pz_ft = 0.5;
end

dataset = 'flickr_both';


% -------------------------------------------------------------------------
%                                load pretrained weights for each pathway
% -------------------------------------------------------------------------

% image pathway
p1 = struct(...
    'typeout', 'step', ...
    'numhid', 1024, ...
    'numstep_h', 5, ...
    'eps', 0.001, ...
    'sp_reg', 1, ...
    'sp_target', 0.2, ...
    'l2reg', 1e-5, ...
    'kcd', 1, ...
    'maxiter', 300, ...
    'batchsize', 100, ...
    'stdinit', 0.005, ...
    'upfactor', 2, ...
    'downfactor', 1);
p2 = struct(...
    'typeout', 'binary', ...
    'numhid', 1024, ...
    'numstep_h', 1, ...
    'eps', 0.1, ...
    'sp_reg', 1, ...
    'sp_target', 0.2, ...
    'l2reg', 1e-5, ...
    'kcd', 1, ...
    'maxiter', 200, ...
    'batchsize', 200, ...
    'stdinit', 0.005, ...
    'upfactor', 2, ...
    'downfactor', 2);

net_img = flickr_img_l2(0, '', 0, ...
    p1.numhid, p1.numstep_h, p1.eps, p1.sp_reg, p1.sp_target, p1.l2reg, p1.kcd, p1.maxiter, p1.batchsize, p1.stdinit, p1.upfactor, p1.downfactor, ...
    p2.numhid, p2.numstep_h, p2.eps, p2.sp_reg, p2.sp_target, p2.l2reg, p2.kcd, p2.maxiter, p2.batchsize, p2.stdinit, p2.upfactor, p2.downfactor);

fprintf('image pathway net loaded!\n');

% text pathway
p1 = struct(...
    'doc_length', 5, ...
    'typein', 'binary', ...
    'typeout', 'binary', ...
    'numhid', 1024, ...
    'numstep_h', 1, ...
    'eps', 0.1, ...
    'sp_reg', 0.01, ...
    'sp_target', 0.2, ...
    'l2reg', 0, ...
    'kcd', 1, ...
    'maxiter', 200, ...
    'batchsize', 100, ...
    'stdinit', 0.005, ...
    'upfactor', 2, ...
    'downfactor', 1);
p2 = struct(...
    'typeout', 'binary', ...
    'numhid', 1024, ...
    'numstep_h', 1, ...
    'eps', 0.03, ...
    'sp_reg', 1, ...
    'sp_target', 0.2, ...
    'l2reg', 1e-5, ...
    'kcd', 1, ...
    'maxiter', 100, ...
    'batchsize', 200, ...
    'stdinit', 0.005, ...
    'upfactor', 2, ...
    'downfactor', 2);

net_text = flickr_text_l2(0, '', 0, p1.doc_length, ...
    p1.numhid, p1.numstep_h, p1.eps, p1.sp_reg, p1.sp_target, p1.l2reg, p1.kcd, p1.maxiter, p1.batchsize, p1.stdinit, p1.upfactor, p1.downfactor, ...
    p2.numhid, p2.numstep_h, p2.eps, p2.sp_reg, p2.sp_target, p2.l2reg, p2.kcd, p2.maxiter, p2.batchsize, p2.stdinit, p2.upfactor, p2.downfactor);

fprintf('text pathway net loaded!\n\n\n');


% load mean and stds for image pathway
[m_global, stds_global] = compute_mean_std(0, 10000);


% -------------------------------------------------------------------------
%                                    train top layer joint representation
% -------------------------------------------------------------------------

params = struct(...
    'dataset', [dataset '_l3'], ...
    'optgpu', optgpu, ...
    'savedir', savedir, ...
    'traintype', traintype, ...
    'doc_length', doc_length, ...
    'imgperbatch', imgperbatch, ...
    'typeinx', typeinx, ...
    'typeinz', typeinz, ...
    'typeout', typeout, ...
    'numvx', net_img{2}.params.numhid, ...
    'numvz', net_text{2}.params.numhid, ...
    'numhid', numhid, ...
    'eps', epsilon, ...
    'eps_decay', eps_decay, ...
    'sp_type', 'exact', ...
    'sp_reg', sp_reg, ...
    'sp_target', sp_target, ...
    'l2reg', l2reg, ...
    'usepcd', 0, ...
    'negchain', batchsize, ...
    'alpha', 1, ...
    'nmf', nmf, ...
    'nstop', nmf, ...
    'draw_sample', 0, ...
    'maxiter', maxiter, ...
    'batchsize', batchsize, ...
    'stdinit', 0.01, ...
    'momentum_change', 5, ...
    'momentum_init', 0.33, ...
    'momentum_final', 0.5, ...
    'upfactor_x', 1, ...
    'upfactor_z', 1, ...
    'downfactor', 2, ...
    'saveiter', 50, ...
    'verbose', 1, ...
    'px', px, ...
    'pz', pz);

[~, params] = mrbm_filename(params);

try
    load(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, params.maxiter), 'weights', 'params');
    fprintf('3rd layer pretrained weights loaded\n\n');
catch
    % load data
    [xunlab, numdim_img] = load_flickr_unlab(doc_length, imgperbatch);
    xunlab_img = xunlab(1:numdim_img, :);
    xunlab_text = xunlab(numdim_img+1:end, :);
    clear xunlab;
    
    % image features (preprocessing and inference)
    xunlab_img = single(xunlab_img);
    xunlab_img = bsxfun(@rdivide, bsxfun(@minus, xunlab_img, m_global), stds_global);
    xunlab_img = net_img{1}.infer(xunlab_img);
    fprintf('done extracting first layer image features\n');
    xunlab_img = net_img{2}.infer(xunlab_img);
    fprintf('done extracting second layer image features\n');
    xunlab_img = single(xunlab_img); % make sure single precision for memory efficiency
    
    % text features (inference)
    xunlab_text = single(xunlab_text);
    xunlab_text = net_text{1}.infer(xunlab_text);
    fprintf('done extracting first layer text features\n');
    xunlab_text = net_text{2}.infer(xunlab_text);
    fprintf('done extracting second layer text features\n');
    xunlab_text = single(xunlab_text); % make sure single precision for memory efficiency
    
    % mrbm training
    [weights, params, history] = mrbm_train(xunlab_img, xunlab_text, params);
    save(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, params.maxiter), 'weights', 'params', 'history');
    
    clear xunlab_img xunlab_text;
end

fname = sprintf('%s_iter_%d', params.fname, maxiter);
infer = mrbm_infer(weights, params, params.nmf);


% -------------------------------------------------------------------------
%                               test with multi-label logistic classifier
% -------------------------------------------------------------------------

if dotest && ~dojoint,
    [xlab, ylab, folds, numdim_img, ~, ~] = load_flickr_lab;
    xlab_img = xlab(1:numdim_img, :);
    xlab_text = xlab(numdim_img+1:end, :);
    clear xlab;
    
    % image features (preprocessing + inference)
    xlab_img = bsxfun(@rdivide, bsxfun(@minus, xlab_img, m_global), stds_global);
    xlab_img = net_img{1}.infer(xlab_img);
    xlab_img = net_img{2}.infer(xlab_img);
    xlab_img = single(xlab_img);
    
    % text features (inference)
    xlab_text = net_text{1}.infer(xlab_text);
    xlab_text = net_text{2}.infer(xlab_text);
    xlab_text = single(xlab_text);
    
    % inference (joint representation)
    xlab_both = infer(xlab_img, xlab_text);
    xlab_img_only = infer(xlab_img, []);
    
    cdataset = params.dataset;
    
    % multimodal (image + text)
    [map_val_mult, map_test_mult] = run_mclr(evalmode, xlab_both, '', ylab, folds, foldlist, l2reg_test, logdir, sprintf('%s_mult', cdataset), fname);
    
    % unimodal (image only)
    [map_val_uni, map_test_uni] = run_mclr(evalmode, xlab_both, xlab_img_only, ylab, folds, foldlist, l2reg_test, logdir, sprintf('%s_uni', cdataset), fname);
    
    clear xlab_both xlab_img_only ylab xlab_img xlab_text;
else
    map_val_mult = -1;
    map_test_mult = -1;
    map_val_uni = -1;
    map_test_uni = -1;
end


% -------------------------------------------------------------------------
%                                                fine-tuning top 2 layers
% -------------------------------------------------------------------------

if dojoint,
    savedir = sprintf('%s/%s', savedir, fname);
    if ~exist(savedir, 'dir'),
        mkdir(savedir);
    end
    
    % initialize parameters and weights
    params = struct(...
        'dataset', [dataset '_l3'], ...
        'traintype', traintype_jt, ...
        'optgpu', optgpu, ...
        'savedir', savedir, ...
        'doc_length', doc_length, ...
        'imgperbatch', imgperbatch, ...
        'typeinx', net_img{2}.params.typein, ...
        'numvx', net_img{2}.params.numvis, ...
        'upfactor_vx', net_img{2}.params.upfactor, ...
        'downfactor_vx', net_img{2}.params.downfactor, ...
        'typeinz', net_text{2}.params.typein, ...
        'numvz', net_text{2}.params.numvis, ...
        'upfactor_vz', net_text{2}.params.upfactor, ...
        'downfactor_vz', net_text{2}.params.downfactor, ...
        'numhx', net_img{2}.params.numhid, ...
        'numhz', net_text{2}.params.numhid, ...
        'numpen', params.numhid, ...
        'eps', epsilon_jt, ...
        'eps_decay', eps_decay_jt, ...
        'sp_type', 'exact', ...
        'sp_target_p', params.sp_target, ...
        'sp_reg_p', sp_reg_jt, ...
        'sp_target_hx', net_img{2}.params.sp_target, ...
        'sp_reg_hx', sp_reg_jt, ...
        'sp_target_hz', net_text{2}.params.sp_target, ...
        'sp_reg_hz', sp_reg_jt, ...
        'l2reg', l2reg_jt, ...
        'nmf', nmf_jt, ...
        'maxiter', maxiter_jt, ...
        'batchsize', batchsize, ...
        'upfactor_x', params.upfactor_x, ...
        'upfactor_z', params.upfactor_z, ...
        'downfactor', 2, ...
        'momentum_change', 5, ...
        'momentum_final', 0.5, ...
        'momentum_init', 0.33, ...
        'sp_damp', 0.9, ...
        'verbose', 1, ...
        'saveiter', 50, ...
        'px', px_jt, ...
        'pz', pz_jt);
    
    [~, params] = mdrnn_l2_filename(params);
    
    % initialize weights
    weights_pre = weights;
    
    weights = struct;
    weights.vxhx = net_img{2}.weights.vishid;
    weights.vzhz = net_text{2}.weights.vishid;
    weights.hxpen = weights_pre.vxhid;
    weights.hzpen = weights_pre.vzhid;
    weights.penbias = weights_pre.hidbias;
    weights.hxbias = net_img{2}.weights.hidbias;
    weights.hxbias = weights.hxbias + weights_pre.vxbias;
    weights.hxbias = weights.hxbias/2;
    weights.hzbias = net_text{2}.weights.hidbias;
    weights.hzbias = weights.hzbias + weights_pre.vzbias;
    weights.hzbias = weights.hzbias/2;
    weights.vxbias = net_img{2}.weights.visbias;
    weights.vzbias = net_text{2}.weights.visbias;
    
    clear weights_pre;
    
    try
        load(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, maxiter_jt), 'weights', 'params');
        fprintf('joint trained weights loaded\n\n');
    catch
        % load data
        [xunlab, numdim_img] = load_flickr_unlab(doc_length, imgperbatch);
        xunlab_img = xunlab(1:numdim_img, :);
        xunlab_text = xunlab(numdim_img+1:end, :);
        clear xunlab;
        
        % image features (preprocessing and inference)
        xunlab_img = single(xunlab_img);
        xunlab_img = bsxfun(@rdivide, bsxfun(@minus, xunlab_img, m_global), stds_global);
        xunlab_img = net_img{1}.infer(xunlab_img);
        fprintf('done extracting first layer image features\n');
        xunlab_img = single(xunlab_img); % make sure single precision for memory efficiency
        
        % text features (inference)
        xunlab_text = single(xunlab_text);
        xunlab_text = net_text{1}.infer(xunlab_text);
        fprintf('done extracting first layer text features\n');
        xunlab_text = single(xunlab_text); % make sure single precision for memory efficiency
        
        [weights, params, history] = mdrnn_l2_train(xunlab_img, xunlab_text, weights, params);
        save(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, maxiter_jt), 'weights', 'params', 'history');
        clear xunlab_img xunlab_text;
    end
    
    infer = mdrnn_l2_infer(weights, params, params.nmf);
    fname = sprintf('%s_iter_%d', params.fname, maxiter_jt);
    
    
    % ---------------------------------------------------------------------
    %                           test with multi-label logistic classifier
    % ---------------------------------------------------------------------
    
    if dotest && ~dofinetune,
        [xlab, ylab, folds, numdim_img, ~, ~] = load_flickr_lab;
        xlab_img = xlab(1:numdim_img, :);
        xlab_text = xlab(numdim_img+1:end, :);
        clear xlab;
        
        % image features (preprocessing + inference)
        xlab_img = bsxfun(@rdivide, bsxfun(@minus, xlab_img, m_global), stds_global);
        xlab_img = net_img{1}.infer(xlab_img);
        xlab_img = single(xlab_img);
        
        % text features (inference)
        xlab_text = net_text{1}.infer(xlab_text);
        xlab_text = single(xlab_text);
        
        % inference (joint representation)
        xlab_both = infer(xlab_img, xlab_text);
        xlab_img_only = infer(xlab_img, []);
        
        cdataset = params.dataset;
        
        % multimodal (image + text)
        [map_val_mult, map_test_mult] = run_mclr(evalmode, xlab_both, '', ylab, folds, foldlist, l2reg_test, logdir, sprintf('%s_mult', cdataset), fname);
        
        % unimodal (image only)
        [map_val_uni, map_test_uni] = run_mclr(evalmode, xlab_both, xlab_img_only, ylab, folds, foldlist, l2reg_test, logdir, sprintf('%s_uni', cdataset), fname);
        
        clear xlab_both xlab_img_only ylab xlab_img xlab_text;
    else
        map_val_mult = -1;
        map_test_mult = -1;
        map_val_uni = -1;
        map_test_uni = -1;
    end
end


% -------------------------------------------------------------------------
%                                               fine-tuning whole network
% -------------------------------------------------------------------------

if dofinetune,
    savedir = sprintf('%s/%s', savedir, fname);
    if ~exist(savedir, 'dir'),
        mkdir(savedir);
    end
    
    % initialize parameters and weights
    params = struct(...
        'dataset', [dataset '_l3'], ...
        'traintype', traintype_ft, ...
        'optgpu', optgpu, ...
        'savedir', savedir, ...
        'doc_length', doc_length, ...
        'imgperbatch', imgperbatch, ...
        'typeinx', net_img{1}.params.typein, ...
        'numvx', net_img{1}.params.numvis, ...
        'upfactor_vx', net_img{1}.params.upfactor, ...
        'downfactor_vx', net_img{1}.params.downfactor, ...
        'typeinz', net_text{1}.params.typein, ...
        'numvz', net_text{1}.params.numvis, ...
        'upfactor_vz', net_text{1}.params.upfactor, ...
        'downfactor_vz', net_text{1}.params.downfactor, ...
        'numhx', net_img{2}.params.numvis, ...
        'upfactor_hx', net_img{2}.params.upfactor, ...
        'downfactor_hx', net_img{2}.params.downfactor, ...
        'numstep_hx', net_img{1}.params.numstep_h, ...
        'numhz', net_text{2}.params.numvis, ...
        'upfactor_hz', net_text{2}.params.upfactor, ...
        'downfactor_hz', net_text{2}.params.downfactor, ...
        'numpx', net_img{2}.params.numhid, ...
        'numpz', net_text{2}.params.numhid, ...
        'numtop', params.numpen, ...
        'eps', epsilon_ft, ...
        'eps_decay', eps_decay_ft, ...
        'sp_type', 'exact', ...
        'sp_target_t', params.sp_target_p, ...
        'sp_reg_t', sp_reg_ft, ...
        'sp_target_px', net_img{2}.params.sp_target, ...
        'sp_reg_px', sp_reg_ft, ...
        'sp_target_pz', net_text{2}.params.sp_target, ...
        'sp_reg_pz', sp_reg_ft, ...
        'sp_target_hx', net_img{1}.params.sp_target, ...
        'sp_reg_hx', sp_reg_ft, ...
        'sp_target_hz', net_text{1}.params.sp_target, ...
        'sp_reg_hz', sp_reg_ft, ...
        'l2reg', l2reg_ft, ...
        'nmf', nmf_ft, ...
        'maxiter', maxiter_ft, ...
        'batchsize', batchsize, ...
        'upfactor_x', params.upfactor_x, ...
        'upfactor_z', params.upfactor_z, ...
        'downfactor', 2, ...
        'momentum_change', 5, ...
        'momentum_final', 0.5, ...
        'momentum_init', 0.33, ...
        'sp_damp', 0.9, ...
        'verbose', 1, ...
        'saveiter', 50, ...
        'px', px_ft, ...
        'pz', pz_ft);
    
    [~, params] = mdrnn_l3_filename(params);
    
    % pretrained weights
    weights_jt = weights;
    
    weights = struct;
    weights.vxhx = net_img{1}.weights.vishid;
    weights.vzhz = net_text{1}.weights.vishid;
    weights.hxpx = weights_jt.vxhx;
    weights.hzpz = weights_jt.vzhz;
    weights.pxtop = weights_jt.hxpen;
    weights.pztop = weights_jt.hzpen;
    weights.topbias = weights_jt.penbias;
    weights.pxbias = weights_jt.hxbias;
    weights.pzbias = weights_jt.hzbias;
    weights.hxbias = net_img{1}.weights.hidbias;
    weights.hxbias = weights.hxbias + weights_jt.vxbias;
    weights.hxbias = weights.hxbias/2;
    weights.hzbias = net_text{1}.weights.hidbias;
    weights.hzbias = weights.hzbias + weights_jt.vzbias;
    weights.hzbias = weights.hzbias/2;
    weights.vxbias = net_img{1}.weights.visbias;
    weights.vzbias = net_text{1}.weights.visbias;
    
    clear weights_jt;
    
    try
        load(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, maxiter_ft), 'weights', 'params');
        fprintf('fine-tuned weights loaded\n\n');
    catch
        % load data
        [xunlab, numdim_img] = load_flickr_unlab(doc_length, imgperbatch);
        xunlab_img = xunlab(1:numdim_img, :);
        xunlab_text = xunlab(numdim_img+1:end, :);
        clear xunlab;
        
        % image features
        xunlab_img = bsxfun(@rdivide, bsxfun(@minus, xunlab_img, m_global), stds_global);
        xunlab_img = single(xunlab_img);
        
        % text features
        xunlab_text = single(xunlab_text);
        fprintf('done extracting input features\n');
        
        [weights, params, history] = mdrnn_l3_train(xunlab_img, xunlab_text, weights, params);
        save(sprintf('%s/%s_iter_%d.mat', savedir, params.fname, maxiter_ft), 'weights', 'params', 'history');
        
        clear xunlab_img xunlab_text;
    end
    
    infer = mdrnn_l3_infer(weights, params, params.nmf, 1);
    fname = sprintf('%s_iter_%d', params.fname, maxiter_ft);
    
    
    % ---------------------------------------------------------------------
    %                           test with multi-label logistic classifier
    % ---------------------------------------------------------------------
    
    if dotest,
        [xlab, ylab, folds, numdim_img, ~, ~] = load_flickr_lab;
        xlab_img = xlab(1:numdim_img, :);
        xlab_text = xlab(numdim_img+1:end, :);
        clear xlab;
        
        % image features (preprocessing + inference)
        xlab_img = bsxfun(@rdivide, bsxfun(@minus, xlab_img, m_global), stds_global);
        xlab_img = single(xlab_img);
        
        % text features (inference)
        xlab_text = single(xlab_text);
        
        % inference (joint representation)
        xlab_both = infer(xlab_img, xlab_text);
        xlab_img_only = infer(xlab_img, []);
        
        cdataset = params.dataset;
        
        % multimodal (image + text)
        [map_val_mult, map_test_mult] = run_mclr(evalmode, xlab_both, '', ylab, folds, foldlist, l2reg_test, logdir, sprintf('%s_mult', cdataset), fname);
        
        % unimodal (image only)
        [map_val_uni, map_test_uni] = run_mclr(evalmode, xlab_both, xlab_img_only, ylab, folds, foldlist, l2reg_test, logdir, sprintf('%s_uni', cdataset), fname);
        
        clear xlab_both xlab_img_only ylab xlab_img xlab_text;
    else
        map_val_mult = -1;
        map_test_mult = -1;
        map_val_uni = -1;
        map_test_uni = -1;
    end
end

return;

