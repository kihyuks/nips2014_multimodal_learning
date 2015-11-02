% -------------------------------------------------
%   Gaussian-binary RBM
%   train stds with gradient descent
%
%   scaled by
%   vishid/std -> vishid
%   visbias/std^2 -> visbias
%   std^3 -> 1
%
%   written by Kihyuk Sohn
% -------------------------------------------------

function [weights, params, grad, history] = rbm_train_real_bin(xtrain, params)


% data preprocessing
if params.normalize,
    xtrain = normalize(xtrain, params.epsnorm);
end

% initialization
weights = struct;
weights.vishid = params.stdinit*randn(params.numvis, params.numhid);
weights.visbias = zeros(params.numvis, 1);
weights.hidbias = zeros(params.numhid, 1);
if params.std_learn,
    if params.std_share,
        weights.stds = sqrt(mean(sum(bsxfun(@minus, xtrain, mean(xtrain, 2)).^2, 2)/size(xtrain, 2)));
    else
        weights.stds = sqrt(sum(bsxfun(@minus, xtrain, mean(xtrain, 2)).^2, 2)/size(xtrain, 2));
    end
else
    weights.stds = ones(params.numvis, 1);
end

% convert to gpu variables
if params.optgpu,
    weights = cpu2gpu_struct(weights);
end

% structs for gradients
grad = replicate_struct(weights, 0);
pos = replicate_struct(weights, 0);
neg = replicate_struct(weights, 0);

% filename to save
fname_mat = sprintf('%s/%s.mat', params.savedir, params.fname_save);
disp(params);


% --------------------------------
% Train gaussian-binary RBM
% --------------------------------

batchsize = params.batchsize;
maxiter = params.maxiter;
runavg_hid = zeros(params.numhid, 1); % for sparsity

% set monitoring variables
history.error = zeros(maxiter,1);
history.sparsity = zeros(maxiter,1);
history.std = zeros(maxiter,1);

if params.usepcd,
    vishid_bu = params.upfactor*bsxfun(@rdivide, weights.vishid, weights.stds);
    hbiasmat = repmat(weights.hidbias, [1, params.negchain]);
    
    negvisprob = repmat(mean(xtrain, 2), [1, params.negchain]);
    negvisstate = negvisprob + bsxfun(@times, weights.stds, randn(size(negvisprob)));
    
    % hidden
    neghidprob = sigmoid(vishid_bu'*negvisstate + hbiasmat);
    neghidstate = sample_bernoulli(neghidprob, params.optgpu);
end

N = size(xtrain, 2);
numbatch = floor(min(N, 100000)/batchsize);

for t = 1:maxiter,
    if t > params.momentum_change,
        momentum = params.momentum_final;
    else
        momentum = params.momentum_init;
    end
    
    epsilon = params.eps/(1+params.eps_decay*t);
    
    recon_err_epoch = zeros(numbatch, 1);
    sparsity_epoch = zeros(numbatch, 1);
    
    randidx = randperm(N);
    
    tS = tic;
    for b = 1:numbatch,
        batchidx = randidx((b-1)*batchsize+1:b*batchsize);
        data = xtrain(:, batchidx);
        if params.optgpu,
            data = gpuArray(single(data));
        end
        
        % compute contrastive divergence steps
        % reshape for speedup
        vishid_bu = params.upfactor*bsxfun(@rdivide, weights.vishid, weights.stds);
        vishid_td = params.downfactor*bsxfun(@times, weights.vishid, weights.stds);
        hbiasmat = repmat(weights.hidbias, [1 batchsize]);
        vbiasmat = repmat(weights.visbias, [1 batchsize]);
        
        % positive phase
        poshidprob = sigmoid(vishid_bu'*data + hbiasmat);
        poshidstate = sample_bernoulli(poshidprob, params.optgpu);
        
        % monitoring variables
        recon = vishid_td*poshidprob + vbiasmat;
        recon_err = sum(sum((recon - data).^2));
        recon_err_epoch(b) = gather(recon_err);
        sparsity_epoch(b) = gather(mean(poshidstate(:)));
        
        
        % negative phase
        if ~params.usepcd,
            neghidstate = poshidstate;
        end
        
        vbiasmat = repmat(weights.visbias, [1 params.negchain]);
        hbiasmat = repmat(weights.hidbias, [1 params.negchain]);
        
        for i = 1:params.kcd,
            % visible
            negvisprob = vishid_td*neghidstate + vbiasmat; % mean of P(v|h)
            negvisstate = negvisprob + bsxfun(@times, weights.stds, randn(size(negvisprob)));
            
            % hidden
            neghidprob = sigmoid(vishid_bu'*negvisstate + hbiasmat);
            neghidstate = sample_bernoulli(neghidprob, params.optgpu);
        end
        
        % visible (for gradient update)
        negvisprob = vishid_td*neghidstate + vbiasmat; % mean of P(v|h)
        
        
        % --------------------------------------------
        % regularizers (sparsity, contrative, l2reg)
        % --------------------------------------------
        
        dh_reg = zeros(size(weights.hidbias));
        dvh_reg = zeros(size(weights.vishid));
        
        % l2 regularzer
        dvh_reg = dvh_reg + params.l2reg*weights.vishid;
        
        % sparsity
        if params.sp_reg > 0,
            if strcmp(params.sp_type, 'exact'),
                hmh = poshidprob.*(1-poshidprob);
                mh = sum(poshidprob,2)/batchsize;
                mhtmp = -params.sp_target./mh + (1-params.sp_target)./(1-mh);
                
                dobj = params.sp_reg*bsxfun(@times, mhtmp, hmh)/batchsize;
                dvh_reg = dvh_reg + data*dobj';
                dh_reg = dh_reg + sum(dobj, 2);
            elseif strcmp(params.sp_type, 'approx'),
                poshidact = mean(poshidprob, 2);
                runavg_hid = params.sp_damp*runavg_hid + (1-params.sp_damp)*poshidact;
                
                dh_reg = dh_reg + params.sp_reg*(runavg_hid - params.sp_target);
                dvh_reg = dvh_reg + params.sp_reg*mean(data, 2)*dh_reg';
            end
        end
        
        
        % gradient (positive , negative)
        pos.vishid = data*poshidprob'/size(data, 2) - dvh_reg;
        pos.hidbias = mean(poshidprob, 2) - dh_reg;
        pos.visbias = mean(data, 2);
        neg.vishid = negvisprob*neghidstate'/size(negvisprob, 2);
        neg.hidbias = mean(neghidstate, 2);
        neg.visbias = mean(negvisprob, 2);
        if params.std_learn,
            pos.stds = mean(data.*(bsxfun(@minus, data - vishid_td*poshidprob,  weights.visbias)), 2);
            neg.stds = weights.stds.^2 - mean(bsxfun(@times, negvisprob, weights.visbias), 2);
            if params.std_share,
                pos.stds = mean(pos.stds);
                neg.stds = mean(neg.stds);
            end
        end
        
        % update parameters
        [weights, grad] = update_params(weights, grad, pos, neg, momentum, epsilon, params.usepcd);
    end
    
    history.error(t) = gather(sum(recon_err_epoch))/numbatch/batchsize;
    history.sparsity(t) = gather(mean(sparsity_epoch));
    history.std(t) = gather(mean(weights.stds));
    
    tE = toc(tS);
    if params.verbose,
        fprintf('epoch %d:\t recon err = %g\t sparsity = %g\t std = %g (time = %g)\n', ...
            t, history.error(t), history.sparsity(t), history.std(t), tE);
    end
    
    if mod(t, params.saveiter) == 0,
        fprintf('epoch %d:\t recon err = %g\t sparsity = %g\t std = %g\n', ...
            t, history.error(t), history.sparsity(t), history.std(t));
        
        % save parameters
        fname_mat_iter = sprintf('%s/%s_iter_%d.mat', params.savedir, params.fname, t);
        save_params(fname_mat_iter, weights, grad, params, t, history);
    end
end

% save parameters
[weights, grad] = save_params(fname_mat, weights, grad, params, t, history);

return;


