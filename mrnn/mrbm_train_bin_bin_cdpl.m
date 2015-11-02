% -------------------------------------------------------------------------
%   binary-binary-binary multimodal RBM
%   trained with CD-percLoss
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


function [weights, params, grad, history] = mrbm_train_bin_bin_cdpl(xtr, ztr, params)


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
%               train binary-binary-binary(hidden) multimodal RBM (minVI)
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

for t = 1:maxiter,
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
        
        data_x = sample_bernoulli(data_x, params.optgpu);
        data_z = sample_bernoulli(data_z, params.optgpu);
        
        % initialize gradient
        pgrad = replicate_struct(pgrad, 0);
        ngrad = replicate_struct(ngrad, 0);
        
        % contrastive divergence
        vxhid_bu = params.upfactor_x*weights.vxhid;
        vzhid_bu = params.upfactor_z*weights.vzhid;
        vxhid_td = params.downfactor*weights.vxhid;
        vzhid_td = params.downfactor*weights.vzhid;
        
        hbiasmat = repmat(weights.hidbias, [1 batchsize]);
        xbiasmat = repmat(weights.vxbias, [1 batchsize]);
        zbiasmat = repmat(weights.vzbias, [1 batchsize]);
        
        
        % positive phase
        wx = vxhid_bu'*data_x;
        wz = vzhid_bu'*data_z;
        poshidprob = sigmoid(wx + wz + hbiasmat);
        
        % monitoring variables
        recon_x = sigmoid(vxhid_td*poshidprob + xbiasmat);
        recon_z = sigmoid(vzhid_td*poshidprob + zbiasmat);
        recon_err_x = sum(sum((recon_x - data_x).^2));
        recon_x_epoch(b) = gather(recon_err_x);
        recon_err_z = sum(sum((recon_z - data_z).^2));
        recon_z_epoch(b) = gather(recon_err_z);
        sparsity_epoch(b) = gather(mean(poshidprob(:)));
        
        % regularizers (sparsity, l2reg)
        dh_reg = zeros(size(weights.hidbias));
        dxh_reg = zeros(size(weights.vxhid));
        dzh_reg = zeros(size(weights.vzhid));
        
        % l2 regularzer
        dxh_reg = dxh_reg + params.l2reg*weights.vxhid;
        dzh_reg = dzh_reg + params.l2reg*weights.vzhid;
        
        % sparsity
        runavg_hid = params.sp_damp*runavg_hid + (1-params.sp_damp)*mean(poshidprob, 2);
        dh_reg = dh_reg + params.sp_reg*(runavg_hid - params.sp_target);
        
        % gradient (positive)
        pgrad.vxhid = pgrad.vxhid + data_x*poshidprob'/size(data_x, 2) - dxh_reg;
        pgrad.vzhid = pgrad.vzhid + data_z*poshidprob'/size(data_z, 2) - dzh_reg;
        pgrad.hidbias = pgrad.hidbias + mean(poshidprob, 2) - dh_reg;
        pgrad.vxbias = pgrad.vxbias + mean(data_x, 2);
        pgrad.vzbias = pgrad.vzbias + mean(data_z, 2);
        
        
        % negative phase
        % P(h,z|x)
        % hidden
        upfactor_x_init = (params.upfactor_x + params.upfactor_z)/params.upfactor_x;
        neghidprob = sigmoid(upfactor_x_init*wx + hbiasmat);
        neghidstate = sample_bernoulli(neghidprob, params.optgpu);
        
        % visible (z)
        negvzprob = sigmoid(vzhid_td*neghidstate + zbiasmat);
        negvzstate = sample_bernoulli(negvzprob, params.optgpu);
        
        fey = mrbm_fey(weights, data_x, negvzstate, [], params);
        negvzstate_best = negvzstate;
        
        for i = 1:params.kcd,
            % hidden
            neghidprob = sigmoid(wx + vzhid_bu'*negvzstate + hbiasmat);
            neghidstate = sample_bernoulli(neghidprob, params.optgpu);
            
            % visible (z)
            negvzprob = sigmoid(vzhid_td*neghidstate + zbiasmat);
            negvzstate = sample_bernoulli(negvzprob, params.optgpu);
            
            cfey = mrbm_fey(weights, data_x, negvzstate, [], params);
            idx = cfey <= fey;
            if sum(idx) > 0,
                fey(idx) = cfey(idx);
                negvzstate_best(:, idx) = negvzstate(:, idx);
            end
        end
        
        % replace with best prediction
        negvzstate = negvzstate_best;
        neghidprob = sigmoid(wx + vzhid_bu'*negvzstate + hbiasmat);
        
        % compute gradient (negative)
        ngrad.vxhid = ngrad.vxhid + 0.5*data_x*neghidprob'/size(data_x, 2);
        ngrad.vzhid = ngrad.vzhid + 0.5*negvzstate*neghidprob'/size(negvzstate, 2);
        ngrad.hidbias = ngrad.hidbias + 0.5*mean(neghidprob, 2);
        ngrad.vxbias = ngrad.vxbias + 0.5*mean(data_x, 2);
        ngrad.vzbias = ngrad.vzbias + 0.5*mean(negvzstate, 2);
        
        
        % P(h,x|z)
        % hidden
        upfactor_z_init = (params.upfactor_x + params.upfactor_z)/params.upfactor_z;
        neghidprob = sigmoid(upfactor_z_init*wz + hbiasmat);
        neghidstate = sample_bernoulli(neghidprob, params.optgpu);
        
        % visible (x)
        negvxprob = sigmoid(vxhid_td*neghidstate + xbiasmat);
        negvxstate = sample_bernoulli(negvxprob, params.optgpu);
        
        fey = mrbm_fey(weights, negvxstate, data_z, [], params);
        negvxstate_best = negvxstate;
        
        for i = 1:params.kcd,
            % hidden
            neghidprob = sigmoid(vxhid_bu'*negvxstate + wz + hbiasmat);
            neghidstate = sample_bernoulli(neghidprob, params.optgpu);
            
            % visible (x)
            negvxprob = sigmoid(vxhid_td*neghidstate + xbiasmat);
            negvxstate = sample_bernoulli(negvxprob, params.optgpu);
            
            cfey = mrbm_fey(weights, negvxstate, data_z, [], params);
            idx = cfey <= fey;
            if sum(idx) > 0,
                fey(idx) = cfey(idx);
                negvxstate_best(:, idx) = negvxstate(:, idx);
            end
        end
        
        % replace with best prediction
        negvxstate = negvxstate_best;
        neghidprob = sigmoid(vxhid_bu'*negvxstate + wz + hbiasmat);
        
        % compute gradient (negative)
        ngrad.vxhid = ngrad.vxhid + 0.5*negvxstate*neghidprob'/size(negvxstate, 2);
        ngrad.vzhid = ngrad.vzhid + 0.5*data_z*neghidprob'/size(data_z, 2);
        ngrad.hidbias = ngrad.hidbias + 0.5*mean(neghidprob, 2);
        ngrad.vxbias = ngrad.vxbias + 0.5*mean(negvxstate, 2);
        ngrad.vzbias = ngrad.vzbias + 0.5*mean(data_z, 2);
        
        
        % -----------------------------------------------------------------
        %                                               update parameters
        % -----------------------------------------------------------------
        
        [weights, grad] = update_params(weights, grad, pgrad, ngrad, momentum, epsilon, 0);
    end
    
    history.error_x(t) = gather(sum(recon_x_epoch))/nbatch/batchsize;
    history.error_z(t) = gather(sum(recon_z_epoch))/nbatch/batchsize;
    history.sparsity(t) = gather(mean(sparsity_epoch));
    
    
    tE = toc(tS);
    if params.verbose,
        fprintf('epoch %d:\t err (x) = %g\t err (z) = %g\t sparsity = %g (time = %g)\n', ...
            t, history.error_x(t), history.error_z(t), history.sparsity(t), tE);
    end
    
    if mod(t, 20) == 0,
        if params.verbose,
            fprintf('epoch %d:\t err (x) = %g\t err (z) = %g\t sparsity = %g\n', ...
                t, history.error_x(t), history.error_z(t), history.sparsity(t));
        end
        
        save_params(fname_mat, weights, grad, params, t, history);
        fprintf('%s\n', fname_mat);
    end
end


% save parameters
[weights, grad] = save_params(fname_mat, weights, grad, params, t, history);


return;


