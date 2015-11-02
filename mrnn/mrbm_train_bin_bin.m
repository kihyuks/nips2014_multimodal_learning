% -------------------------------------------------------------------------
%   binary-binary-binary multimodal RBM
%   objective function: log P(x,z)
%
%   xtr  : numvx x numimg
%   ztr  : numvz x numimg
%
%   params
%   weights
%   grad
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------


function [weights, params, grad, history] = mrbm_train_bin_bin(xtr, ztr, params)


% -------------------------------------------------------------------------
%                                                          initialization
% -------------------------------------------------------------------------

% weight initialization
weights = struct;
weights.vxhid = params.stdinit*randn(params.numvx, params.numhid);
weights.vzhid = params.stdinit*randn(params.numvz, params.numhid);
weights.vxbias = arcsigm(clip(mean(xtr, 2)));
weights.vzbias = arcsigm(clip(mean(ztr, 2)));
weights.hidbias = zeros(params.numhid, 1);

% convert to gpu variables
if params.optgpu,
    weights = cpu2gpu_struct(weights);
end

% structs for gradients
grad = replicate_struct(weights, 0);
pgrad = replicate_struct(weights, 0);
ngrad = replicate_struct(weights, 0);

% filename to save
fname_mat = sprintf('%s/%s.mat', params.savedir, params.fname_save);
disp(params);


% -------------------------------------------------------------------------
%                  train binary-binary-binary(hidden) multimodal RBM (ML)
% -------------------------------------------------------------------------

batchsize = params.batchsize;
maxiter = params.maxiter;
runavg_hid = zeros(params.numhid, 1); % for sparsity

% set monitoring variables
history.error_x = zeros(maxiter,1);
history.error_z = zeros(maxiter,1);
history.sparsity = zeros(maxiter,1);

nexample = size(xtr, 2);
nbatch = floor(min(nexample, 100000)/batchsize);

% train with pcd
if params.usepcd,
    vxhid_bu = params.upfactor_x*weights.vxhid;
    vzhid_bu = params.upfactor_z*weights.vzhid;
    
    hbiasmat = repmat(weights.hidbias, [1, params.negchain]);
    
    % visible (x)
    negvxprob = repmat(sigmoid(weights.vxbias), [1, params.negchain]);
    negvxstate = sample_bernoulli(negvxprob, params.optgpu);
    
    % visible (z)
    negvzprob = repmat(sigmoid(weights.vzbias), [1, params.negchain]);
    negvzstate = sample_bernoulli(negvzprob, params.optgpu);
    
    % hidden
    neghidprob = sigmoid(vxhid_bu'*negvxstate + vzhid_bu'*negvzstate + hbiasmat);
    neghidstate = sample_bernoulli(neghidprob, params.optgpu);
end


for t = 1:maxiter,
    % gradually increase the number of CD-steps
    kcd = min(1 + floor((t-1)/10), params.kcd);
    
    if t > params.momentum_change,
        momentum = params.momentum_final;
    else
        momentum = params.momentum_init;
    end
    
    epsilon = params.eps/(1+params.eps_decay*t);
    
    recon_x_epoch = zeros(nbatch, 1);
    recon_z_epoch = zeros(nbatch, 1);
    sparsity_epoch = zeros(nbatch, 1);
    
    randidx = randperm(nexample);
    
    tS = tic;
    for b = 1:nbatch,
        batchidx = randidx((b-1)*batchsize+1:b*batchsize);
        data_x = xtr(:, batchidx);
        data_z = ztr(:, batchidx);
        
        % draw sample
        data_x = sample_bernoulli(data_x, params.optgpu);
        data_z = sample_bernoulli(data_z, params.optgpu);
        
        % contrastive divergence
        vxhid_bu = params.upfactor_x*weights.vxhid;
        vxhid_td = params.downfactor*weights.vxhid;
        vzhid_bu = params.upfactor_z*weights.vzhid;
        vzhid_td = params.downfactor*weights.vzhid;
        
        hbiasmat = repmat(weights.hidbias, [1 batchsize]);
        xbiasmat = repmat(weights.vxbias, [1 batchsize]);
        zbiasmat = repmat(weights.vzbias, [1 batchsize]);
        
        
        % positive phase
        poshidprob = sigmoid(vxhid_bu'*data_x + vzhid_bu'*data_z + hbiasmat);
        poshidstate = sample_bernoulli(poshidprob, params.optgpu);
        
        % monitoring variables
        recon_x = sigmoid(vxhid_td*poshidprob + xbiasmat);
        recon_err_x = sum(sum((recon_x - data_x).^2));
        recon_x_epoch(b) = gather(recon_err_x);
        recon_z = sigmoid(vzhid_td*poshidprob + zbiasmat);
        recon_err_z = sum(sum((recon_z - data_z).^2));
        recon_z_epoch(b) = gather(recon_err_z);
        sparsity_epoch(b) = gather(mean(poshidprob(:)));
        
        
        % negative phase
        if ~params.usepcd,
            neghidstate = poshidstate;
        end
        
        hbiasmat = repmat(weights.hidbias, [1 params.negchain]);
        xbiasmat = repmat(weights.vxbias, [1 params.negchain]);
        zbiasmat = repmat(weights.vzbias, [1 params.negchain]);
        
        for i = 1:kcd,
            % visible (x)
            negvxprob = sigmoid(vxhid_td*neghidstate + xbiasmat);
            negvxstate = sample_bernoulli(negvxprob, params.optgpu);
            
            % visible (z)
            negvzprob = sigmoid(vzhid_td*neghidstate + zbiasmat);
            negvzstate = sample_bernoulli(negvzprob, params.optgpu);
            
            % hidden
            neghidprob = sigmoid(vxhid_bu'*negvxstate + vzhid_bu'*negvzstate + hbiasmat);
            neghidstate = sample_bernoulli(neghidprob, params.optgpu);
        end
        negvxprob = sigmoid(vxhid_td*neghidstate + xbiasmat);
        negvzprob = sigmoid(vzhid_td*neghidstate + zbiasmat);
        
        % gradient (positive , negative)
        pgrad.vxhid = data_x*poshidprob'/size(data_x, 2);
        pgrad.vzhid = data_z*poshidprob'/size(data_z, 2);
        pgrad.hidbias = mean(poshidprob, 2);
        pgrad.vxbias = mean(data_x, 2);
        pgrad.vzbias = mean(data_z, 2);
        
        ngrad.vxhid = negvxprob*neghidstate'/size(negvxprob, 2);
        ngrad.vzhid = negvzprob*neghidstate'/size(negvzprob, 2);
        ngrad.hidbias = mean(neghidstate, 2);
        ngrad.vxbias = mean(negvxprob, 2);
        ngrad.vzbias = mean(negvzprob, 2);
        
        
        % -----------------------------------------------------------------
        %                                  regularizers (sparsity, l2reg)
        % -----------------------------------------------------------------
        
        dh_reg = zeros(size(weights.hidbias));
        dxh_reg = zeros(size(weights.vxhid));
        dzh_reg = zeros(size(weights.vzhid));
        
        % l2 weight decaying
        dxh_reg = dxh_reg + params.l2reg*weights.vxhid;
        dzh_reg = dzh_reg + params.l2reg*weights.vzhid;
        
        % sparsity
        if params.sp_reg > 0,
            poshidact = mean(poshidprob, 2);
            runavg_hid = params.sp_damp*runavg_hid + (1-params.sp_damp)*poshidact;
            
            dh_reg = dh_reg + params.sp_reg*(runavg_hid - params.sp_target);
        end
        
        % gradient
        pgrad.vxhid = pgrad.vxhid - dxh_reg;
        pgrad.vzhid = pgrad.vzhid - dzh_reg;
        pgrad.hidbias = pgrad.hidbias - dh_reg;
        
        
        % -----------------------------------------------------------------
        %                                               update parameters
        % -----------------------------------------------------------------
        
        [weights, grad] = update_params(weights, grad, pgrad, ngrad, momentum, epsilon, params.usepcd);
    end
    
    history.error_x(t) = gather(sum(recon_x_epoch))/nbatch/batchsize;
    history.error_z(t) = gather(sum(recon_z_epoch))/nbatch/batchsize;
    history.sparsity(t) = gather(mean(sparsity_epoch));
    
    tE = toc(tS);
    if params.verbose,
        fprintf('epoch %d (kcd = %d):\t err (x) = %g\t err (z) = %g\t sparsity = %g (time = %g)\n', ...
            t, kcd, history.error_x(t), history.error_z(t), history.sparsity(t), tE);
    end
    
    if mod(t, 20) == 0,
        if params.verbose,
            fprintf('epoch %d (kcd = %d):\t err (x) = %g\t err (z) = %g\t sparsity = %g\n', ...
                t, kcd, history.error_x(t), history.error_z(t), history.sparsity(t));
        end
        
        % save parameters
        save_params(sprintf('%s/%s_iter_%d.mat', params.savedir, params.fname, t), weights, grad, params, t, history);
        fprintf('%s\n', fname_mat);
    end
end


% save parameters and visualizations
[weights, grad] = save_params(fname_mat, weights, grad, params, t, history);


return;


