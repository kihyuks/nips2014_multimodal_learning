% -------------------------------------------------------------------------
%   binary-binary-binary multimodal RBM
%   rnn backprop + joint log-likelihood
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


function [weights, params, grad, history] = mrnn_train_bin_bin(xtr, ztr, params)


% -------------------------------------------------------------------------
%                                                          initialization
% -------------------------------------------------------------------------

rng('default')

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
history.sparsity_x = zeros(maxiter,1);
history.error_z = zeros(maxiter,1);
history.sparsity_z = zeros(maxiter,1);

nexample = size(xtr, 2);
nbatch = min(floor(min(nexample, 100000)/batchsize), 500);

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

% # mean-field iterations
nmf = params.nmf;

for t = 1:maxiter,
    % gradually increase the backprop over epoch
    nstop = min(params.nstop, 1+floor((t-1)/20));
    kcd = min(params.kcd, 1+floor((t-1)/10));
    
    fprintf('[%d/%d] nmf = %d, nstop = %d\n', t, maxiter, nmf, nstop);
    
    if t > params.momentum_change,
        momentum = params.momentum_final;
    else
        momentum = params.momentum_init;
    end
    
    epsilon = params.eps/(1+params.eps_decay*t);
    
    recon_x_epoch = zeros(nbatch, 1);
    sparsity_x_epoch = zeros(nbatch, 1);
    recon_z_epoch = zeros(nbatch, 1);
    sparsity_z_epoch = zeros(nbatch, 1);
    
    randidx = randperm(nexample);
    
    tS = tic;
    
    hidprob = zeros(params.numhid, batchsize, nmf+1);
    vxprob = zeros(params.numvx, batchsize, nmf+1);
    vzprob = zeros(params.numvz, batchsize, nmf+1);
    if params.optgpu,
        hidprob = gpuArray(single(hidprob));
        vxprob = gpuArray(single(vxprob));
        vzprob = gpuArray(single(vzprob));
    end
    
    for b = 1:nbatch,
        batchidx = randidx((b-1)*batchsize+1:b*batchsize);
        data_x = xtr(:, batchidx);
        data_z = ztr(:, batchidx);
        
        % initialize gradient
        pgrad = replicate_struct(pgrad, 0);
        ngrad = replicate_struct(ngrad, 0);
        
        % contrastive divergence
        vxhid_bu = params.upfactor_x*weights.vxhid;
        vxhid_td = params.downfactor*weights.vxhid;
        vzhid_bu = params.upfactor_z*weights.vzhid;
        vzhid_td = params.downfactor*weights.vzhid;
        
        hbiasmat = repmat(weights.hidbias, [1 batchsize]);
        xbiasmat = repmat(weights.vxbias, [1 batchsize]);
        zbiasmat = repmat(weights.vzbias, [1 batchsize]);
        
        % corrupt input data with rate < params.px or pz
        mask_x = sample_bernoulli(params.px*rand*ones(size(data_x)), params.optgpu);
        mask_z = sample_bernoulli(params.pz*rand*ones(size(data_z)), params.optgpu);
        data_x_corrupt = data_x.*mask_x;
        data_z_corrupt = data_z.*mask_z;
        
        
        % -----------------------------------------------------------------
        % P(h,z|x)
        % -----------------------------------------------------------------
        
        wx = vxhid_bu'*data_x;
        
        % hidden
        upfactor_x_init = (params.upfactor_x + params.upfactor_z - params.upfactor_z*params.pz)/params.upfactor_x;
        hidprob(:, :, 1) = sigmoid(upfactor_x_init*wx + vzhid_bu'*data_z_corrupt + hbiasmat);
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
        
        % monitoring variables
        recon_err_z = sum(sum((vzprob(:, :, nmf+1) - data_z).^2));
        recon_z_epoch(b) = gather(recon_err_z);
        sparsity_z_epoch(b) = gather(mean(hidprob(:)));
        
        % compute gradient (full backprop clamped by 1)
        dobj = params.alpha*(data_z - vzprob(:, :, nmf+1))/batchsize;
        
        % vz -> hid
        pgrad.vzhid = pgrad.vzhid + params.downfactor*dobj*hidprob(:, :, nmf+1)';
        pgrad.vzbias = pgrad.vzbias + sum(dobj, 2);
        hmh = hidprob(:, :, nmf+1).*(1-hidprob(:, :, nmf+1));
        dobj = hmh.*(vzhid_td'*dobj);
        
        % exact sparsity
        if params.sp_reg > 0,
            mh = mean(hidprob(:, :, nmf+1),2);
            mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
            mhtmp = params.sp_target./mh - (1-params.sp_target)./(1-mh);
            dobj = dobj + params.alpha*params.sp_reg*bsxfun(@times, mhtmp, hmh)/batchsize;
        end
        
        % hid -> vx, bias
        if nmf == 0,
            pgrad.vxhid = pgrad.vxhid + params.upfactor_x*upfactor_x_init*data_x*dobj';
            pgrad.vzhid = pgrad.vzhid + params.upfactor_z*data_z_corrupt*dobj';
            pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
        else
            pgrad.vxhid = pgrad.vxhid + params.upfactor_x*data_x*dobj';
            pgrad.vzhid = pgrad.vzhid + params.upfactor_z*vzprob(:, :, nmf)*dobj';
            pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
            dobj = (1-mask_z).*vzprob(:, :, nmf).*(1-vzprob(:, :, nmf)).*(vzhid_bu*dobj);
        end
        
        for i = nmf:-1:nmf-nstop+1,
            if i == 1,
                % vz -> hid
                pgrad.vzhid = pgrad.vzhid + params.downfactor*dobj*hidprob(:, :, i)';
                pgrad.vzbias = pgrad.vzbias + sum(dobj, 2);
                dobj = hidprob(:, :, i).*(1-hidprob(:, :, i)).*(vzhid_td'*dobj);
                
                % hid -> vx, vz, bias
                pgrad.vxhid = pgrad.vxhid + params.upfactor_x*upfactor_x_init*data_x*dobj';
                pgrad.vzhid = pgrad.vzhid + params.upfactor_z*data_z_corrupt*dobj';
                pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
            else
                % vz -> hid
                pgrad.vzhid = pgrad.vzhid + params.downfactor*dobj*hidprob(:, :, i)';
                pgrad.vzbias = pgrad.vzbias + sum(dobj, 2);
                dobj = hidprob(:, :, i).*(1-hidprob(:, :, i)).*(vzhid_td'*dobj);
                
                % hid -> vx, bias
                pgrad.vxhid = pgrad.vxhid + params.upfactor_x*data_x*dobj';
                pgrad.vzhid = pgrad.vzhid + params.upfactor_z*vzprob(:, :, i-1)*dobj';
                pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
                dobj = (1-mask_z).*vzprob(:, :, i-1).*(1-vzprob(:, :, i-1)).*(vzhid_bu*dobj);
            end
        end
        
        
        % -----------------------------------------------------------------
        % P(h,x|z)
        % -----------------------------------------------------------------
        
        wz = vzhid_bu'*data_z;
        
        % hidden
        upfactor_z_init = (params.upfactor_z + params.upfactor_x - params.upfactor_x*params.px)/params.upfactor_z;
        hidprob(:, :, 1) = sigmoid(vxhid_bu'*data_x_corrupt + upfactor_z_init*wz + hbiasmat);
        for i = 1:nmf,
            % visible (x)
            vxprob(:, :, i) = sigmoid(vxhid_td*hidprob(:, :, i) + xbiasmat);
            vxprob(:, :, i) = vxprob(:, :, i).*(1-mask_x) + data_x_corrupt;
            
            % hidden
            hidprob(:, :, i+1) = sigmoid(wz + vxhid_bu'*vxprob(:, :, i) + hbiasmat);
        end
        % visible (x)
        vxprob(:, :, nmf+1) = sigmoid(vxhid_td*hidprob(:, :, nmf+1) + xbiasmat);
        vxprob(:, :, nmf+1) = vxprob(:, :, nmf+1).*(1-mask_x) + data_x_corrupt;
        
        % monitoring variables
        recon_err_x = sum(sum((vxprob(:, :, nmf+1) - data_x).^2));
        recon_x_epoch(b) = gather(recon_err_x);
        sparsity_x_epoch(b) = gather(mean(hidprob(:)));
        
        % compute gradient (full backprop clamped by 1)
        dobj = params.alpha*(data_x - vxprob(:, :, nmf+1))/batchsize;
        
        % vx -> hid
        pgrad.vxhid = pgrad.vxhid + params.downfactor*dobj*hidprob(:, :, nmf+1)';
        pgrad.vxbias = pgrad.vxbias + sum(dobj, 2);
        hmh = hidprob(:, :, nmf+1).*(1-hidprob(:, :, nmf+1));
        dobj = hmh.*(vxhid_td'*dobj);
        
        % exact sparsity
        if params.sp_reg > 0,
            mh = mean(hidprob(:, :, nmf+1),2);
            mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
            mhtmp = params.sp_target./mh - (1-params.sp_target)./(1-mh);
            dobj = dobj + params.alpha*params.sp_reg*bsxfun(@times, mhtmp, hmh)/batchsize;
        end
        
        % hid -> vz, bias
        if nmf == 0,
            pgrad.vzhid = pgrad.vzhid + params.upfactor_z*upfactor_z_init*data_z*dobj';
            pgrad.vxhid = pgrad.vxhid + params.upfactor_x*data_x_corrupt*dobj';
            pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
        else
            pgrad.vzhid = pgrad.vzhid + params.upfactor_z*data_z*dobj';
            pgrad.vxhid = pgrad.vxhid + params.upfactor_x*vxprob(:, :, nmf)*dobj';
            pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
            dobj = (1-mask_x).*vxprob(:, :, nmf).*(1-vxprob(:, :, nmf)).*(vxhid_bu*dobj);
        end
        
        for i = nmf:-1:nmf-nstop+1,
            if i == 1,
                % vx -> hid
                pgrad.vxhid = pgrad.vxhid + params.downfactor*dobj*hidprob(:, :, i)';
                pgrad.vxbias = pgrad.vxbias + sum(dobj, 2);
                dobj = hidprob(:, :, i).*(1-hidprob(:, :, i)).*(vxhid_td'*dobj);
                
                % hid -> vx, vz, bias
                pgrad.vzhid = pgrad.vzhid + params.upfactor_z*upfactor_z_init*data_z*dobj';
                pgrad.vxhid = pgrad.vxhid + params.upfactor_x*data_x_corrupt*dobj';
                pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
            else
                % vx -> hid
                pgrad.vxhid = pgrad.vxhid + params.downfactor*dobj*hidprob(:, :, i)';
                pgrad.vxbias = pgrad.vxbias + sum(dobj, 2);
                dobj = hidprob(:, :, i).*(1-hidprob(:, :, i)).*(vxhid_td'*dobj);
                
                % hid -> vx, vz, bias
                pgrad.vzhid = pgrad.vzhid + params.upfactor_z*data_z*dobj';
                pgrad.vxhid = pgrad.vxhid + params.upfactor_x*vxprob(:, :, i-1)*dobj';
                pgrad.hidbias = pgrad.hidbias + sum(dobj, 2);
                dobj = (1-mask_x).*vxprob(:, :, i-1).*(1-vxprob(:, :, i-1)).*(vxhid_bu*dobj);
            end
        end
        
        
        % -----------------------------------------------------------------
        %             gradient for joint log-likelihood of multimodal RBM
        % -----------------------------------------------------------------
        
        if params.alpha < 1,
            % draw sample
            data_x = sample_bernoulli(data_x, params.optgpu);
            data_z = sample_bernoulli(data_z, params.optgpu);
            
            % positive phase
            poshidprob = sigmoid(vxhid_bu'*data_x + vzhid_bu'*data_z + hbiasmat);
            poshidstate = sample_bernoulli(poshidprob, params.optgpu);
            
            dh_reg = zeros(size(weights.hidbias));
            if params.sp_reg > 0,
                poshidact = mean(poshidprob, 2);
                runavg_hid = params.sp_damp*runavg_hid + (1-params.sp_damp)*poshidact;
                
                dh_reg = dh_reg + (1-params.alpha)*params.sp_reg*(runavg_hid - params.sp_target);
            end
            
            % negative phase
            if ~params.usepcd,
                neghidstate = poshidstate;
            end
            
            hbiasmat = repmat(weights.hidbias, [1 params.negchain]);
            xbiasmat = repmat(weights.vxbias, [1 params.negchain]);
            zbiasmat = repmat(weights.vzbias, [1 params.negchain]);
            
            for i = 1:kcd,
                % visible
                negvxprob = sigmoid(vxhid_td*neghidstate + xbiasmat);
                negvxstate = sample_bernoulli(negvxprob, params.optgpu);
                
                negvzprob = sigmoid(vzhid_td*neghidstate + zbiasmat);
                negvzstate = sample_bernoulli(negvzprob, params.optgpu);
                
                % hidden
                neghidprob = sigmoid(vxhid_bu'*negvxstate + vzhid_bu'*negvzstate + hbiasmat);
                neghidstate = sample_bernoulli(neghidprob, params.optgpu);
            end
            negvxprob = sigmoid(vxhid_td*neghidstate + xbiasmat);
            negvzprob = sigmoid(vzhid_td*neghidstate + zbiasmat);
            
            % gradient (positive , negative)
            pgrad.vxhid = pgrad.vxhid + (1-params.alpha)*data_x*poshidprob'/size(data_x, 2);
            pgrad.vzhid = pgrad.vzhid + (1-params.alpha)*data_z*poshidprob'/size(data_z, 2);
            pgrad.hidbias = pgrad.hidbias + (1-params.alpha)*mean(poshidprob, 2) - dh_reg;
            pgrad.vxbias = pgrad.vxbias + (1-params.alpha)*mean(data_x, 2);
            pgrad.vzbias = pgrad.vzbias + (1-params.alpha)*mean(data_z, 2);
            
            ngrad.vxhid = ngrad.vxhid + (1-params.alpha)*negvxprob*neghidstate'/size(negvxprob, 2);
            ngrad.vzhid = ngrad.vzhid + (1-params.alpha)*negvzprob*neghidstate'/size(negvzprob, 2);
            ngrad.hidbias = ngrad.hidbias + (1-params.alpha)*mean(neghidstate, 2);
            ngrad.vxbias = ngrad.vxbias + (1-params.alpha)*mean(negvxprob, 2);
            ngrad.vzbias = ngrad.vzbias + (1-params.alpha)*mean(negvzprob, 2);
        end
        
        
        % -----------------------------------------------------------------
        %                                             regularizer (l2reg)
        % -----------------------------------------------------------------
        
        % l2 weigth decaying
        pgrad.vxhid = pgrad.vxhid - params.l2reg*weights.vxhid;
        pgrad.vzhid = pgrad.vzhid - params.l2reg*weights.vzhid;
        
        
        % -----------------------------------------------------------------
        %                                               update parameters
        % -----------------------------------------------------------------
        
        [weights, grad, flag] = update_params(weights, grad, pgrad, ngrad, momentum, epsilon, 0);
        if flag,
            % temporarily reduce epsilon for the current epoch
            epsilon = 0.95*epsilon;
        end
    end
    history.error_x(t) = gather(sum(recon_x_epoch))/nbatch/batchsize;
    history.sparsity_x(t) = gather(mean(sparsity_x_epoch));
    history.error_z(t) = gather(sum(recon_z_epoch))/nbatch/batchsize;
    history.sparsity_z(t) = gather(mean(sparsity_z_epoch));
    
    tE = toc(tS);
    if params.verbose,
        fprintf('epoch %d:\t recon err (x) = %g\t recon err (z) = %g\t sp(x) = %g\t sp(z) = %g (time = %g)\n', ...
            t, history.error_x(t), history.error_z(t), history.sparsity_x(t), history.sparsity_z(t), tE);
    end
    
    
    % save parameters (saveiter) epoch
    if mod(t, params.saveiter) == 0,
        fprintf('epoch %d:\t recon err (x) = %g\t recon err (z) = %g\t sp(x) = %g\t sp(z) = %g\n', ...
            t, history.error_x(t), history.error_z(t), history.sparsity_x(t), history.sparsity_z(t));
        
        params_at_save = params;
        params_at_save.nmf = nmf;
        params_at_save.nstop = nstop;
        save_params(sprintf('%s/%s_iter_%d.mat', params.savedir, params.fname, t), weights, grad, params_at_save, t, history);
        
        fprintf('%s\n', fname_mat);
    end
end


% save parameters
params.nmf = nmf;
params.nstop = nstop;
[weights, grad] = save_params(fname_mat, weights, grad, params, t, history);


return;
