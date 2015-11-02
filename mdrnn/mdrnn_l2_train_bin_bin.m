% -------------------------------------------------------------------------
%   binary-binary-binary multimodal RNN (2layers)
%   x -> z, z -> x
%
%   xtr  : numvx x numimg
%   ztr  : numvz x numimg
%
%   params
%   weights (initialized from pretrained network)
%   grad
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------


function [weights, params, grad, history] = mdrnn_l2_train_bin_bin(xtr, ztr, weights, params)

rng('default')

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
%               train binary-binary-binary(hidden) multimodal RNN (minVI)
% -------------------------------------------------------------------------

batchsize = params.batchsize;
maxiter = params.maxiter;
runavg_hx = zeros(params.numhx, 1);
runavg_hz = zeros(params.numhz, 1);
runavg_pen = zeros(params.numpen, 1); % for sparsity

% set monitoring variables
history.error_x = zeros(maxiter,1);
history.error_z = zeros(maxiter,1);
history.sparsity_x = zeros(maxiter,1);
history.sparsity_z = zeros(maxiter,1);
history.sparsity = zeros(maxiter,1);

nexample = size(xtr, 2);
nbatch = min(floor(min(nexample, 100000)/batchsize), 500);

% # mean-field iterations
nmf = params.nmf;

for t = 1:maxiter,
    if t > params.momentum_change,
        momentum = params.momentum_final;
    else
        momentum = params.momentum_init;
    end
    
    epsilon = params.eps/(1+params.eps_decay*t);
    
    recon_err_x_epoch = zeros(nbatch, 1);
    recon_err_z_epoch = zeros(nbatch, 1);
    sparsity_x_epoch = zeros(nbatch, 1);
    sparsity_z_epoch = zeros(nbatch, 1);
    sparsity_epoch = zeros(nbatch, 1);
    
    randidx = randperm(nexample);
    
    tS = tic;
    
    penprob = zeros(params.numpen, batchsize, nmf+1);
    hxprob_td = zeros(params.numhx, batchsize, nmf+1);
    hzprob_td = zeros(params.numhz, batchsize, nmf+1);
    if nmf > 0,
        hxprob_bu = zeros(params.numhx, batchsize, nmf);
        hzprob_bu = zeros(params.numhz, batchsize, nmf);
    end
    vxprob = zeros(params.numvx, batchsize, nmf+1);
    vzprob = zeros(params.numvz, batchsize, nmf+1);
    
    if params.optgpu,
        penprob = gpuArray(single(penprob));
        hxprob_td = gpuArray(single(hxprob_td));
        hzprob_td = gpuArray(single(hzprob_td));
        if nmf > 0,
            hxprob_bu = gpuArray(single(hxprob_bu));
            hzprob_bu = gpuArray(single(hzprob_bu));
        end
        vxprob = gpuArray(single(vxprob));
        vzprob = gpuArray(single(vzprob));
    end
    
    for b = 1:nbatch,
        batchidx = randidx((b-1)*batchsize+1:b*batchsize);
        data_x = xtr(:, batchidx);
        data_z = ztr(:, batchidx);
        
        % initialize gradient
        pgrad = replicate_struct(pgrad, 0);
        
        % reshape for speedup
        vxhx_bu = params.upfactor_vx*weights.vxhx;
        vxhx_td = params.downfactor_vx*weights.vxhx;
        vzhz_bu = params.upfactor_vz*weights.vzhz;
        vzhz_td = params.downfactor_vz*weights.vzhz;
        hxpen_bu = params.upfactor_x*weights.hxpen;
        hxpen_td = params.downfactor*weights.hxpen;
        hzpen_bu = params.upfactor_z*weights.hzpen;
        hzpen_td = params.downfactor*weights.hzpen;
        
        pbiasmat = repmat(weights.penbias, [1 batchsize]);
        hxbiasmat = repmat(weights.hxbias, [1 batchsize]);
        hzbiasmat = repmat(weights.hzbias, [1 batchsize]);
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
        
        hxprob = sigmoid(vxhx_bu'*data_x + hxbiasmat);
        whx = hxpen_bu'*hxprob;
        
        hzprob = sigmoid(vzhz_bu'*data_z_corrupt + hzbiasmat);
        whz = hzpen_bu'*hzprob;
        
        % hidden
        upfactor_x_init = (params.upfactor_x + params.upfactor_z*(1-params.pz))/params.upfactor_x;
        upfactor_z_init = (params.upfactor_z*params.pz)/params.upfactor_z;
        penprob(:, :, 1) = sigmoid(upfactor_x_init*whx + upfactor_z_init*whz + pbiasmat);
        for i = 1:nmf,
            % hidden (z)
            hzprob_td(:, :, i) = sigmoid(hzpen_td*penprob(:, :, i) + hzbiasmat);
            
            % visible (z)
            vzprob(:, :, i) = sigmoid(vzhz_td*hzprob_td(:, :, i) + zbiasmat);
            vzprob(:, :, i) = vzprob(:, :, i).*(1-mask_z) + data_z_corrupt;
            
            % hidden (z)
            hzprob_bu(:, :, i) = sigmoid(vzhz_bu'*vzprob(:, :, i) + hzbiasmat);
            
            % hidden
            penprob(:, :, i+1) = sigmoid(whx + hzpen_bu'*hzprob_bu(:, :, i) + pbiasmat);
        end
        % hidden (z)
        hzprob_td(:, :, nmf+1) = sigmoid(hzpen_td*penprob(:, :, nmf+1) + hzbiasmat);
        
        % visible (z)
        vzprob(:, :, nmf+1) = sigmoid(vzhz_td*hzprob_td(:, :, nmf+1) + zbiasmat);
        vzprob(:, :, nmf+1) = vzprob(:, :, nmf+1).*(1-mask_z) + data_z_corrupt;
        
        % for sparsity penalty
        poshxact = mean(hxprob, 2);
        if nmf == 0,
            poshzact = mean(hzprob_td(:, :, nmf+1), 2);
        else
            poshzact = mean(hzprob_bu(:, :, nmf), 2);
        end
        pospenact = mean(penprob(:, :, nmf+1), 2);
        
        runavg_pen = params.sp_damp*runavg_pen + (1-params.sp_damp)*pospenact;
        runavg_hx = params.sp_damp*runavg_hx + (1-params.sp_damp)*poshxact;
        runavg_hz = params.sp_damp*runavg_hz + (1-params.sp_damp)*poshzact;
        
        % monitoring variables
        recon_err_z = sum(sum((vzprob(:, :, nmf+1) - data_z).^2));
        recon_err_z_epoch(b) = gather(recon_err_z);
        
        sparsity_z_epoch(b) = gather(mean(poshzact));
        sparsity_epoch(b) = gather(mean(pospenact));
        
        
        % some variables
        hxh = hxprob.*(1-hxprob);
        
        % compute gradient (full backprop)
        dobj = (data_z - vzprob(:, :, nmf+1))/batchsize;
        
        for i = nmf+1:-1:1,
            % vz -> hz_td
            pgrad.vzhz = pgrad.vzhz + params.downfactor_vz*dobj*hzprob_td(:, :, i)';
            pgrad.vzbias = pgrad.vzbias + sum(dobj, 2);
            dobj = hzprob_td(:, :, i).*(1-hzprob_td(:, :, i)).*(vzhz_td'*dobj);
            
            % hz_td -> pen
            pgrad.hzpen = pgrad.hzpen + params.downfactor*dobj*penprob(:, :, i)';
            pgrad.hzbias = pgrad.hzbias + sum(dobj, 2);
            hmh = penprob(:, :, i).*(1-penprob(:, :, i));
            dobj = hmh.*(hzpen_td'*dobj);
            
            % sparsity (pen)
            if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_p > 0,
                mh = sum(penprob(:, :, i),2)/batchsize;
                mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
                mhtmp = params.sp_target_p./mh - (1-params.sp_target_p)./(1-mh);
                dobj = dobj + params.sp_reg_p*bsxfun(@times, mhtmp, hmh)/batchsize;
            end
            
            if i == 1,
                % pen -> hx
                pgrad.hxpen = pgrad.hxpen + params.upfactor_x*upfactor_x_init*hxprob*dobj';
                pgrad.penbias = pgrad.penbias + sum(dobj, 2);
                dobj_x = hxh.*(upfactor_x_init*hxpen_bu*dobj);
                
                % hx -> vx
                pgrad.vxhx = pgrad.vxhx + params.upfactor_vx*data_x*dobj_x';
                pgrad.hxbias = pgrad.hxbias + sum(dobj_x, 2);
                
                % pen -> hz_bu
                pgrad.hzpen = pgrad.hzpen + params.upfactor_z*upfactor_z_init*hzprob*dobj';
                dobj = hzprob.*(1-hzprob).*(upfactor_z_init*hzpen_bu*dobj);
                
                % hz_bu -> vz
                pgrad.vzhz = pgrad.vzhz + params.upfactor_vz*data_z_corrupt*dobj';
                pgrad.hzbias = pgrad.hzbias + sum(dobj, 2);
            else
                % pen -> hx
                pgrad.hxpen = pgrad.hxpen + params.upfactor_x*hxprob*dobj';
                pgrad.penbias = pgrad.penbias + sum(dobj, 2);
                dobj_x = hxh.*(hxpen_bu*dobj);
                
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_hx > 0,
                    hmh = hxh;
                    mh = sum(hxprob,2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
                    mhtmp = params.sp_target_hx./mh - (1-params.sp_target_hx)./(1-mh);
                    dobj_x = dobj_x + params.sp_reg_hx*bsxfun(@times, mhtmp, hmh)/batchsize;
                end
                
                % hx -> vx
                pgrad.vxhx = pgrad.vxhx + params.upfactor_vx*data_x*dobj_x';
                pgrad.hxbias = pgrad.hxbias + sum(dobj_x, 2);
                
                % pen -> hz_bu
                pgrad.hzpen = pgrad.hzpen + params.upfactor_z*hzprob_bu(:, :, i-1)*dobj';
                hmh = hzprob_bu(:, :, i-1).*(1-hzprob_bu(:, :, i-1));
                dobj = hmh.*(hzpen_bu*dobj);
                
                % sparsity (hz_bu)
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_hz > 0,
                    mh = sum(hzprob_bu(:, :, i-1),2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
                    mhtmp = params.sp_target_hz./mh - (1-params.sp_target_hz)./(1-mh);
                    dobj = dobj + params.sp_reg_hz*bsxfun(@times, mhtmp, hmh)/batchsize;
                end
                
                % hz_bu -> vz
                pgrad.vzhz = pgrad.vzhz + params.upfactor_vz*vzprob(:, :, i-1)*dobj';
                pgrad.hzbias = pgrad.hzbias + sum(dobj, 2);
                dobj = (1-mask_z).*vzprob(:, :, i-1).*(1-vzprob(:, :, i-1)).*(vzhz_bu*dobj);
            end
        end
        
        
        % -----------------------------------------------------------------
        % P(h,x|z)
        % -----------------------------------------------------------------
        
        hzprob = sigmoid(vzhz_bu'*data_z + hzbiasmat);
        whz = hzpen_bu'*hzprob;
        
        hxprob = sigmoid(vxhx_bu'*data_x_corrupt + hxbiasmat);
        whx = hxpen_bu'*hxprob;
        
        % hidden
        upfactor_z_init = (params.upfactor_z + params.upfactor_x*(1-params.px))/params.upfactor_z;
        upfactor_x_init = (params.upfactor_x*params.px)/params.upfactor_x;
        penprob(:, :, 1) = sigmoid(upfactor_x_init*whx + upfactor_z_init*whz + pbiasmat);
        for i = 1:nmf,
            % hidden (x)
            hxprob_td(:, :, i) = sigmoid(hxpen_td*penprob(:, :, i) + hxbiasmat);
            
            % visible (x)
            vxprob(:, :, i) = sigmoid(vxhx_td*hxprob_td(:, :, i) + xbiasmat);
            vxprob(:, :, i) = vxprob(:, :, i).*(1-mask_x) + data_x_corrupt;
            
            % hidden (x)
            hxprob_bu(:, :, i) = sigmoid(vxhx_bu'*vxprob(:, :, i) + hxbiasmat);
            
            % hidden
            penprob(:, :, i+1) = sigmoid(whz + hxpen_bu'*hxprob_bu(:, :, i) + pbiasmat);
        end
        % hidden (x)
        hxprob_td(:, :, nmf+1) = sigmoid(hxpen_td*penprob(:, :, nmf+1) + hxbiasmat);
        
        % visible (x)
        vxprob(:, :, nmf+1) = sigmoid(vxhx_td*hxprob_td(:, :, nmf+1) + xbiasmat);
        vxprob(:, :, nmf+1) = vxprob(:, :, nmf+1).*(1-mask_x) + data_x_corrupt;
        
        % for sparsity penalty
        poshzact = mean(hzprob, 2);
        if nmf == 0,
            poshxact = mean(hxprob_td(:, :, nmf+1), 2);
        else
            poshxact = mean(hxprob_bu(:, :, nmf), 2);
        end
        pospenact = mean(penprob(:, :, nmf+1), 2);
        
        runavg_pen = params.sp_damp*runavg_pen + (1-params.sp_damp)*pospenact;
        runavg_hz = params.sp_damp*runavg_hz + (1-params.sp_damp)*poshzact;
        runavg_hx = params.sp_damp*runavg_hx + (1-params.sp_damp)*poshxact;
        
        % monitoring variables
        recon_err_x = sum(sum((vxprob(:, :, nmf+1) - data_x).^2));
        recon_err_x_epoch(b) = gather(recon_err_x);
        
        sparsity_x_epoch(b) = gather(mean(poshxact));
        sparsity_epoch(b) = sparsity_epoch(b) + gather(mean(pospenact));
        sparsity_epoch(b) = sparsity_epoch(b)/2;
        
        % some variables
        hzh = hzprob.*(1-hzprob);
        
        % compute gradient (full backprop)
        dobj = (data_x - vxprob(:, :, nmf+1))/batchsize;
        
        for i = nmf+1:-1:1,
            % vx -> hx_td
            pgrad.vxhx = pgrad.vxhx + params.downfactor_vx*dobj*hxprob_td(:, :, i)';
            pgrad.vxbias = pgrad.vxbias + sum(dobj, 2);
            dobj = hxprob_td(:, :, i).*(1-hxprob_td(:, :, i)).*(vxhx_td'*dobj);
            
            % hx_td -> pen
            pgrad.hxpen = pgrad.hxpen + params.downfactor*dobj*penprob(:, :, i)';
            pgrad.hxbias = pgrad.hxbias + sum(dobj, 2);
            hmh = penprob(:, :, i).*(1-penprob(:, :, i));
            dobj = hmh.*(hxpen_td'*dobj);
            
            % sparsity (pen)
            if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_p > 0,
                mh = sum(penprob(:, :, i),2)/batchsize;
                mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
                mhtmp = params.sp_target_p./mh - (1-params.sp_target_p)./(1-mh);
                dobj = dobj + params.sp_reg_p*bsxfun(@times, mhtmp, hmh)/batchsize;
            end
            
            if i == 1,
                % pen -> hz
                pgrad.hzpen = pgrad.hzpen + params.upfactor_z*upfactor_z_init*hzprob*dobj';
                pgrad.penbias = pgrad.penbias + sum(dobj, 2);
                dobj_z = hzh.*(upfactor_z_init*hzpen_bu*dobj);
                
                % hz -> vz
                pgrad.vzhz = pgrad.vzhz + params.upfactor_vz*data_z*dobj_z';
                pgrad.hzbias = pgrad.hzbias + sum(dobj_z, 2);
                
                % pen -> hx_bu
                pgrad.hxpen = pgrad.hxpen + params.upfactor_x*upfactor_x_init*hxprob*dobj';
                dobj = hxprob.*(1-hxprob).*(upfactor_x_init*hxpen_bu*dobj);
                
                % hx_bu -> vx
                pgrad.vxhx = pgrad.vxhx + params.upfactor_vx*data_x_corrupt*dobj';
                pgrad.hxbias = pgrad.hxbias + sum(dobj, 2);
            else
                % pen -> hz
                pgrad.hzpen = pgrad.hzpen + params.upfactor_z*hzprob*dobj';
                pgrad.penbias = pgrad.penbias + sum(dobj, 2);
                dobj_z = hzh.*(hzpen_bu*dobj);
                
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_hz > 0,
                    hmh = hzh;
                    mh = sum(hzprob,2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
                    mhtmp = params.sp_target_hz./mh - (1-params.sp_target_hz)./(1-mh);
                    dobj_z = dobj_z + params.sp_reg_hz*bsxfun(@times, mhtmp, hmh)/batchsize;
                end
                
                % hz -> vz
                pgrad.vzhz = pgrad.vzhz + params.upfactor_vz*data_z*dobj_z';
                pgrad.hzbias = pgrad.hzbias + sum(dobj_z, 2);
                
                % pen -> hx_bu
                pgrad.hxpen = pgrad.hxpen + params.upfactor_x*hxprob_bu(:, :, i-1)*dobj';
                hmh = hxprob_bu(:, :, i-1).*(1-hxprob_bu(:, :, i-1));
                dobj = hmh.*(hxpen_bu*dobj);
                
                % sparsity (hx_bu)
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_hx > 0,
                    mh = sum(hxprob_bu(:, :, i-1),2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6); % for numerical stability
                    mhtmp = params.sp_target_hx./mh - (1-params.sp_target_hx)./(1-mh);
                    dobj = dobj + params.sp_reg_hx*bsxfun(@times, mhtmp, hmh)/batchsize;
                end
                
                % hx_bu -> vx
                pgrad.vxhx = pgrad.vxhx + params.upfactor_vx*vxprob(:, :, i-1)*dobj';
                pgrad.hxbias = pgrad.hxbias + sum(dobj, 2);
                dobj = (1-mask_x).*vxprob(:, :, i-1).*(1-vxprob(:, :, i-1)).*(vxhx_bu*dobj);
            end
        end
        
        
        % -----------------------------------------------------------------
        %                                      regularizer (l2, sparsity)
        % -----------------------------------------------------------------
        
        dhx_reg = zeros(size(weights.hxbias));
        dhz_reg = zeros(size(weights.hzbias));
        dp_reg = zeros(size(weights.penbias));
        dvhx_reg = zeros(size(weights.vxhx));
        dvhz_reg = zeros(size(weights.vzhz));
        dhxp_reg = zeros(size(weights.hxpen));
        dhzp_reg = zeros(size(weights.hzpen));
        
        % l2 regularzer
        dvhx_reg = dvhx_reg + params.l2reg*weights.vxhx;
        dvhz_reg = dvhz_reg + params.l2reg*weights.vzhz;
        dhxp_reg = dhxp_reg + params.l2reg*weights.hxpen;
        dhzp_reg = dhzp_reg + params.l2reg*weights.hzpen;
        
        % sparsity
        if strcmp(params.sp_type, 'approx'),
            dp_reg = dp_reg + params.sp_reg_p*(runavg_pen - params.sp_target_p);
            dhx_reg = dhx_reg + params.sp_reg_hx*(runavg_hx - params.sp_target_hx);
            dhz_reg = dhz_reg + params.sp_reg_hz*(runavg_hz - params.sp_target_hz);
        end
        
        % gradient
        pgrad.vxhx = pgrad.vxhx - dvhx_reg;
        pgrad.vzhz = pgrad.vzhz - dvhz_reg;
        pgrad.hxpen = pgrad.hxpen - dhxp_reg;
        pgrad.hzpen = pgrad.hzpen - dhzp_reg;
        pgrad.penbias = pgrad.penbias - dp_reg;
        pgrad.hxbias = pgrad.hxbias - dhx_reg;
        pgrad.hzbias = pgrad.hzbias - dhz_reg;
        
        
        % -----------------------------------------------------------------
        %                                               update parameters
        % -----------------------------------------------------------------
        
        [weights, grad, flag] = update_params(weights, grad, pgrad, ngrad, momentum, epsilon, 0);
        if flag,
            % temporarily reduce epsilon for the current epoch
            epsilon = 0.95*epsilon;
        end
    end
    
    history.error_x(t) = gather(sum(recon_err_x_epoch))/nbatch/batchsize;
    history.error_z(t) = gather(sum(recon_err_z_epoch))/nbatch/batchsize;
    history.sparsity_x(t) = gather(mean(sparsity_x_epoch));
    history.sparsity_z(t) = gather(mean(sparsity_z_epoch));
    history.sparsity(t) = gather(mean(sparsity_epoch));
    
    tE = toc(tS);
    if params.verbose,
        fprintf('epoch %d:\t err (x) = %g\t err (z) = %g\t sp (x) = %g\t sp (z) = %g\t sp (pen) = %g (time = %g)\n', ...
            t, history.error_x(t), history.error_z(t), history.sparsity_x(t), history.sparsity_z(t), history.sparsity(t), tE);
    end
    
    % save parameters every few epochs
    if mod(t, params.saveiter) == 0,
        fprintf('epoch %d:\t err (x) = %g\t err (z) = %g\t sp (x) = %g\t sp (z) = %g\t sp (pen) = %g\n', ...
            t, history.error_x(t), history.error_z(t), history.sparsity_x(t), history.sparsity_z(t), history.sparsity(t));
        
        params_at_save = params;
        params_at_save.nmf = nmf;
        save_params(sprintf('%s/%s_iter_%d.mat', params.savedir, params.fname, t), weights, grad, params_at_save, t, history);
        
        fprintf('%s/%s_iter_%d.mat\n', params.savedir, params.fname, t);
    end
end

% save parameters
params.nmf = nmf;
[weights, grad] = save_params(fname_mat, weights, grad, params, t, history);

return;
