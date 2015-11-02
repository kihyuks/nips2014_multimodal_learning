% -------------------------------------------------------------------------
%   (real/binary)-binary-binary multimodal RNN (3layers)
%   x -> z
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


function [weights, params, grad, history] = mdrnn_l3_train_bin_xtoz(xtr, ztr, weights, params)

rng('default');

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
%        train (real/binary)-binary-binary(hidden) multimodal RNN (minVI)
% -------------------------------------------------------------------------

batchsize = params.batchsize;
maxiter = params.maxiter;

% for sparsity
runavg_hx = zeros(params.numhx, 1);
runavg_hz = zeros(params.numhz, 1);
runavg_px = zeros(params.numpx, 1);
runavg_pz = zeros(params.numpz, 1);
runavg_top = zeros(params.numtop, 1);

% set monitoring variables
history.error_z = zeros(maxiter,1);
history.sparsity_hx = zeros(maxiter,1);
history.sparsity_hz = zeros(maxiter,1);
history.sparsity_px = zeros(maxiter,1);
history.sparsity_pz = zeros(maxiter,1);
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
    
    recon_err_z_epoch = zeros(nbatch, 1);
    sparsity_hx_epoch = zeros(nbatch, 1);
    sparsity_hz_epoch = zeros(nbatch, 1);
    sparsity_px_epoch = zeros(nbatch, 1);
    sparsity_pz_epoch = zeros(nbatch, 1);
    sparsity_epoch = zeros(nbatch, 1);
    
    randidx = randperm(nexample);
    
    tS = tic;
    
    topprob = zeros(params.numtop, batchsize, nmf+1);
    pzprob_td = zeros(params.numpz, batchsize, nmf+1);
    pzprob_bu = zeros(params.numpz, batchsize, nmf);
    hzprob_td = zeros(params.numhz, batchsize, nmf+1);
    hzprob_bu = zeros(params.numhz, batchsize, nmf);
    vzprob = zeros(params.numvz, batchsize, nmf+1);
    
    if params.optgpu,
        topprob = gpuArray(single(topprob));
        pzprob_td = gpuArray(single(pzprob_td));
        hzprob_td = gpuArray(single(hzprob_td));
        vzprob = gpuArray(single(vzprob));
        if nmf > 0,
            pzprob_bu = gpuArray(single(pzprob_bu));
            hzprob_bu = gpuArray(single(hzprob_bu));
        end
    end
    
    for b = 1:nbatch,
        batchidx = randidx((b-1)*batchsize+1:b*batchsize);
        data_x = xtr(:, batchidx);
        data_z = ztr(:, batchidx);
        
        % initialize gradient
        pgrad = replicate_struct(pgrad, 0);
        
        % reshape for speedup
        vxhx_bu = params.upfactor_vx*weights.vxhx;
        vzhz_bu = params.upfactor_vz*weights.vzhz;
        vzhz_td = params.downfactor_vz*weights.vzhz;
        hxpx_bu = params.upfactor_hx*weights.hxpx;
        hzpz_bu = params.upfactor_hz*weights.hzpz;
        hzpz_td = params.downfactor_hz*weights.hzpz;
        pxtop_bu = params.upfactor_x*weights.pxtop;
        pztop_bu = params.upfactor_z*weights.pztop;
        pztop_td = params.downfactor*weights.pztop;
        
        tbiasmat = repmat(weights.topbias, [1 batchsize]);
        pxbiasmat = repmat(weights.pxbias, [1 batchsize]);
        pzbiasmat = repmat(weights.pzbias, [1 batchsize]);
        hxbiasmat = repmat(weights.hxbias, [1 batchsize]);
        hzbiasmat = repmat(weights.hzbias, [1 batchsize]);
        zbiasmat = repmat(weights.vzbias, [1 batchsize]);
        
        % corrupt input data
        mask_z = sample_bernoulli(params.pz*rand*ones(size(zbiasmat)), params.optgpu);
        data_z_corrupt = data_z.*mask_z;
        
        
        % -----------------------------------------------------------------
        % P(h,x|z)
        % -----------------------------------------------------------------
        
        hxprob = sigmoid(vxhx_bu'*data_x + hxbiasmat);
        pxprob = sigmoid(hxpx_bu'*hxprob + pxbiasmat);
        wpx = pxtop_bu'*pxprob;
        
        hzprob = sigmoid(vzhz_bu'*data_z_corrupt + hzbiasmat);
        pzprob = sigmoid(hzpz_bu'*hzprob + pzbiasmat);
        wpz = pztop_bu'*pzprob;
        
        % hidden
        upfactor_x_init = (params.upfactor_x + params.upfactor_z*(1-params.pz))/params.upfactor_x;
        upfactor_z_init = (params.upfactor_z*params.pz)/params.upfactor_z;
        topprob(:, :, 1) = sigmoid(upfactor_x_init*wpx + upfactor_z_init*wpz + tbiasmat);
        for i = 1:nmf,
            % pen (z)
            pzprob_td(:, :, i) = sigmoid(pztop_td*topprob(:, :, i) + pzbiasmat);
            
            % hidden (z)
            hzprob_td(:, :, i) = sigmoid(hzpz_td*pzprob_td(:, :, i) + hzbiasmat);
            
            % visible (z)
            vzprob(:, :, i) = sigmoid(vzhz_td*hzprob_td(:, :, i) + zbiasmat);
            vzprob(:, :, i) = vzprob(:, :, i).*(1-mask_z) + data_z_corrupt;
            
            % hidden (z)
            hzprob_bu(:, :, i) = sigmoid(vzhz_bu'*vzprob(:, :, i) + hzbiasmat);
            
            % pen
            pzprob_bu(:, :, i) = sigmoid(hzpz_bu'*hzprob_bu(:, :, i) + pzbiasmat);
            
            % hidden
            topprob(:, :, i+1) = sigmoid(wpx + pztop_bu'*pzprob_bu(:, :, i) + tbiasmat);
        end
        % pen (z)
        pzprob_td(:, :, nmf+1) = sigmoid(pztop_td*topprob(:, :, nmf+1) + pzbiasmat);
        
        % hidden (z)
        hzprob_td(:, :, nmf+1) = sigmoid(hzpz_td*pzprob_td(:, :, nmf+1) + hzbiasmat);
        
        % visible (z)
        vzprob(:, :, nmf+1) = sigmoid(vzhz_td*hzprob_td(:, :, nmf+1) + zbiasmat);
        vzprob(:, :, nmf+1) = vzprob(:, :, nmf+1).*(1-mask_z) + data_z_corrupt;
        
        % for sparsity penalty
        poshxact = mean(hxprob, 2);
        pospxact = mean(pxprob, 2);
        if nmf == 0,
            poshzact = mean(hzprob_td(:, :, nmf+1), 2);
        else
            poshzact = mean(hzprob_bu(:, :, nmf), 2);
        end
        if nmf == 0,
            pospzact = mean(pzprob_td(:, :, nmf+1), 2);
        else
            pospzact = mean(pzprob_bu(:, :, nmf), 2);
        end
        postopact = mean(topprob(:, :, nmf+1), 2);
        
        runavg_top = params.sp_damp*runavg_top + (1-params.sp_damp)*postopact;
        runavg_px = params.sp_damp*runavg_px + (1-params.sp_damp)*pospxact;
        runavg_pz = params.sp_damp*runavg_pz + (1-params.sp_damp)*pospzact;
        runavg_hx = params.sp_damp*runavg_hx + (1-params.sp_damp)*poshxact;
        runavg_hz = params.sp_damp*runavg_hz + (1-params.sp_damp)*poshzact;
        
        % monitoring variables
        recon_err_z = sum(sum((vzprob(:, :, nmf+1) - data_z).^2));
        recon_err_z_epoch(b) = gather(recon_err_z);
        
        sparsity_hx_epoch(b) = gather(mean(poshxact));
        sparsity_hz_epoch(b) = gather(mean(poshzact));
        sparsity_px_epoch(b) = gather(mean(pospxact));
        sparsity_pz_epoch(b) = gather(mean(pospzact));
        sparsity_epoch(b) = gather(mean(postopact));
        
        
        % some variables
        pxp = pxprob.*(1-pxprob);
        hxh = hxprob.*(1-hxprob);
        pzp = pzprob.*(1-pzprob);
        hzh = hzprob.*(1-hzprob);
        
        % compute gradient (full backprop)
        dobj = (data_z - vzprob(:, :, nmf+1))/batchsize;
        
        for i = nmf+1:-1:1,
            % vz -> hz_td
            pgrad.vzhz = pgrad.vzhz + params.downfactor_vz*dobj*hzprob_td(:, :, i)';
            pgrad.vzbias = pgrad.vzbias + sum(dobj, 2);
            dobj = hzprob_td(:, :, i).*(1-hzprob_td(:, :, i)).*(vzhz_td'*dobj);
            
            % hz_td -> pz_td
            pgrad.hzpz = pgrad.hzpz + params.downfactor_hz*dobj*pzprob_td(:, :, i)';
            pgrad.hzbias = pgrad.hzbias + sum(dobj, 2);
            dobj = pzprob_td(:, :, i).*(1-pzprob_td(:, :, i)).*(hzpz_td'*dobj);
            
            % pz_td -> top
            pgrad.pztop = pgrad.pztop + params.downfactor*dobj*topprob(:, :, i)';
            pgrad.pzbias = pgrad.pzbias + sum(dobj, 2);
            hmh = topprob(:, :, i).*(1-topprob(:, :, i));
            dobj = hmh.*(pztop_td'*dobj);
            
            % sparsity (top)
            if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_t > 0,
                mh = sum(topprob(:, :, i),2)/batchsize;
                mh = max(min(mh, 1-1e-6), 1e-6);
                mhtmp = params.sp_target_t./mh - (1-params.sp_target_t)./(1-mh);
                dobj = dobj + params.sp_reg_t*bsxfun(@times, mhtmp, hmh)/batchsize;
            end
            
            if i == 1,
                % top -> px
                pgrad.pxtop = pgrad.pxtop + params.upfactor_x*upfactor_x_init*pxprob*dobj';
                pgrad.pztop = pgrad.pztop + params.upfactor_z*upfactor_z_init*pzprob*dobj';
                pgrad.topbias = pgrad.topbias + sum(dobj, 2);
                dobj_x = pxp.*(upfactor_x_init*pxtop_bu*dobj);
                dobj = pzp.*(upfactor_z_init*pztop_bu*dobj);
                
                % px -> hx
                pgrad.hxpx = pgrad.hxpx + params.upfactor_hx*hxprob*dobj_x';
                pgrad.pxbias = pgrad.pxbias + sum(dobj_x, 2);
                hmh = hxh;
                dobj_x = hmh.*(hxpx_bu*dobj_x);
                
                % hx -> vx
                pgrad.vxhx = pgrad.vxhx + params.upfactor_vx*data_x*dobj_x';
                pgrad.hxbias = pgrad.hxbias + sum(dobj_x, 2);
                
                % pz_bu -> hz_bu
                pgrad.hzpz = pgrad.hzpz + params.upfactor_hz*hzprob*dobj';
                pgrad.pzbias = pgrad.pzbias + sum(dobj, 2);
                dobj = hzh.*(hzpz_bu*dobj);
                
                % hz_bu -> vz
                pgrad.vzhz = pgrad.vzhz + params.upfactor_vz*data_z_corrupt*dobj';
                pgrad.hzbias = pgrad.hzbias + sum(dobj, 2);
            else
                % top -> px
                pgrad.pxtop = pgrad.pxtop + params.upfactor_x*pxprob*dobj';
                pgrad.topbias = pgrad.topbias + sum(dobj, 2);
                hmh = pxp;
                dobj_x = hmh.*(pxtop_bu*dobj);
                
                % sparsity (px)
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_px > 0,
                    mh = sum(pxprob,2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6);
                    mhtmp = params.sp_target_px./mh - (1-params.sp_target_px)./(1-mh);
                    dobj_x = dobj_x + params.sp_reg_px*bsxfun(@times, mhtmp, hmh)/batchsize;
                end
                
                % px -> hx
                pgrad.hxpx = pgrad.hxpx + params.upfactor_hx*hxprob*dobj_x';
                pgrad.pxbias = pgrad.pxbias + sum(dobj_x, 2);
                hmh = hxh;
                dobj_x = hmh.*(hxpx_bu*dobj_x);
                
                % sparsity (hx)
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_hx > 0,
                    mh = sum(hxprob,2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6);
                    mhtmp = params.sp_target_hx./mh - (1-params.sp_target_hx)./(1-mh);
                    dobj_x = dobj_x + params.sp_reg_hx*bsxfun(@times, mhtmp, hmh)/batchsize;
                end
                
                % hx -> vx
                pgrad.vxhx = pgrad.vxhx + params.upfactor_vx*data_x*dobj_x';
                pgrad.hxbias = pgrad.hxbias + sum(dobj_x, 2);
                
                % top -> pz_bu
                pgrad.pztop = pgrad.pztop + params.upfactor_z*pzprob_bu(:, :, i-1)*dobj';
                hmh = pzprob_bu(:, :, i-1).*(1-pzprob_bu(:, :, i-1));
                dobj = hmh.*(pztop_bu*dobj);
                
                % sparsity (pz_bu)
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_pz > 0,
                    mh = sum(pzprob_bu(:, :, i-1),2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6);
                    mhtmp = params.sp_target_pz./mh - (1-params.sp_target_pz)./(1-mh);
                    dobj = dobj + params.sp_reg_pz*bsxfun(@times, mhtmp, hmh)/batchsize;
                end
                
                % pz_bu -> hz_bu
                pgrad.hzpz = pgrad.hzpz + params.upfactor_hz*hzprob_bu(:, :, i-1)*dobj';
                pgrad.pzbias = pgrad.pzbias + sum(dobj, 2);
                hmh = hzprob_bu(:, :, i-1).*(1-hzprob_bu(:, :, i-1));
                dobj = hmh.*(hzpz_bu*dobj);
                
                % sparsity (hz_bu)
                if i == nmf+1 && strcmp(params.sp_type, 'exact') && params.sp_reg_hz > 0,
                    mh = sum(hzprob_bu(:, :, i-1),2)/batchsize;
                    mh = max(min(mh, 1-1e-6), 1e-6);
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
        %                                      regularizer (l2, sparsity)
        % -----------------------------------------------------------------
        
        dhx_reg = zeros(size(weights.hxbias));
        dhz_reg = zeros(size(weights.hzbias));
        dpx_reg = zeros(size(weights.pxbias));
        dpz_reg = zeros(size(weights.pzbias));
        dt_reg = zeros(size(weights.topbias));
        dvhx_reg = zeros(size(weights.vxhx));
        dvhz_reg = zeros(size(weights.vzhz));
        dhpx_reg = zeros(size(weights.hxpx));
        dhpz_reg = zeros(size(weights.hzpz));
        dpxt_reg = zeros(size(weights.pxtop));
        dpzt_reg = zeros(size(weights.pztop));
        
        % l2 regularizer
        dvhx_reg = dvhx_reg + params.l2reg*weights.vxhx;
        dvhz_reg = dvhz_reg + params.l2reg*weights.vzhz;
        dhpx_reg = dhpx_reg + params.l2reg*weights.hxpx;
        dhpz_reg = dhpz_reg + params.l2reg*weights.hzpz;
        dpxt_reg = dpxt_reg + params.l2reg*weights.pxtop;
        dpzt_reg = dpzt_reg + params.l2reg*weights.pztop;
        
        % sparsity
        if strcmp(params.sp_type, 'approx'),
            dt_reg = dt_reg + params.sp_reg_t*(runavg_top - params.sp_target_t);
            dpx_reg = dpx_reg + params.sp_reg_px*(runavg_px - params.sp_target_px);
            dpz_reg = dpz_reg + params.sp_reg_pz*(runavg_pz - params.sp_target_pz);
            dhx_reg = dhx_reg + params.sp_reg_hx*(runavg_hx - params.sp_target_hx);
            dhz_reg = dhz_reg + params.sp_reg_hz*(runavg_hz - params.sp_target_hz);
        end
        
        % gradient
        pgrad.vxhx = pgrad.vxhx - dvhx_reg;
        pgrad.vzhz = pgrad.vzhz - dvhz_reg;
        pgrad.hxpx = pgrad.hxpx - dhpx_reg;
        pgrad.hzpz = pgrad.hzpz - dhpz_reg;
        pgrad.pxtop = pgrad.pxtop - dpxt_reg;
        pgrad.pztop = pgrad.pztop - dpzt_reg;
        pgrad.topbias = pgrad.topbias - dt_reg;
        pgrad.pxbias = pgrad.pxbias - dpx_reg;
        pgrad.pzbias = pgrad.pzbias - dpz_reg;
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
    
    history.error_z(t) = gather(sum(recon_err_z_epoch))/nbatch/batchsize;
    history.sparsity_hx(t) = gather(mean(sparsity_hx_epoch));
    history.sparsity_hz(t) = gather(mean(sparsity_hz_epoch));
    history.sparsity_px(t) = gather(mean(sparsity_px_epoch));
    history.sparsity_pz(t) = gather(mean(sparsity_pz_epoch));
    history.sparsity(t) = gather(mean(sparsity_epoch));
    
    tE = toc(tS);
    if params.verbose,
        fprintf('epoch %d:\t err (z) = %g\t sp (x) = %g\t sp (z) = %g\t sp (pen) = %g (time = %g)\n', ...
            t, history.error_z(t), history.sparsity_px(t), history.sparsity_pz(t), history.sparsity(t), tE);
    end
    
    % save parameters every few epochs
    if mod(t, params.saveiter) == 0,
        fprintf('epoch %d:\t err (z) = %g\t sp (x) = %g\t sp (z) = %g\t sp (pen) = %g\n', ...
            t, history.error_z(t), history.sparsity_px(t), history.sparsity_pz(t), history.sparsity(t));
        
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
