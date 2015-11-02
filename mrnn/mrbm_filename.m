function [fname, params] = mrbm_filename(params)

switch params.traintype,
    case 'pcd',
        % MRBM with LL using PCD training (baseline)
        params.nmf = 0;
        params.nstop = 0;
        params.sp_damp = 0.9;
        params.sp_type = 'approx';
        fname = sprintf('%s_%s_x_%d_z_%d_h_%d_eps_%g_%g_%s_target_%g_reg_%g_l2r_%g_bs_%d_init_%g_up_%d_%d_down_%d_mom_%d_pcd_%d_%d_neg_%d', ...
            params.dataset, params.traintype, params.numvx, params.numvz, params.numhid, params.eps, ...
            params.eps_decay, params.sp_type, params.sp_target, params.sp_reg, params.l2reg, ...
            params.batchsize, params.stdinit, params.upfactor_x, params.upfactor_z, ...
            params.downfactor, params.momentum_change, params.usepcd, params.kcd, params.negchain);
        
    case 'cdpl',
        % MRBM with VI using CD training (proposed)
        params.nmf = 0;
        params.nstop = 0;
        params.usepcd = 0;
        params.sp_damp = 0.9;
        params.sp_type = 'approx';
        fname = sprintf('%s_%s_x_%d_z_%d_h_%d_eps_%g_%g_%s_target_%g_reg_%g_l2r_%g_bs_%d_init_%g_up_%d_%d_down_%d_mom_%d_kcd_%d', ...
            params.dataset, params.traintype, params.numvx, params.numvz, params.numhid, params.eps, ...
            params.eps_decay, params.sp_type, params.sp_target, params.sp_reg, params.l2reg, ...
            params.batchsize, params.stdinit, params.upfactor_x, params.upfactor_z, ...
            params.downfactor, params.momentum_change, params.kcd);
        
    case 'mrnn',
        % MRBM with VI using MP training (proposed)
        params.usepcd = 0;
        params.kcd = 0;
        fname = sprintf('%s_%s_x_%d_z_%d_h_%d_eps_%g_%g_%s_target_%g_reg_%g_l2r_%g_bs_%d_init_%g_up_%d_%d_down_%d_mom_%d_nmf_%d_px_%g_pz_%g', ...
            params.dataset, params.traintype, params.numvx, params.numvz, params.numhid, params.eps, ...
            params.eps_decay, params.sp_type, params.sp_target, params.sp_reg, params.l2reg, ...
            params.batchsize, params.stdinit, params.upfactor_x, params.upfactor_z, ...
            params.downfactor, params.momentum_change, params.nmf, params.px, params.pz);
        
    case 'hybrid',
        % MRBM with VI using MP training + joint log-likelihood with PCD (proposed)
        if params.alpha == 1,
            params.traintype = 'mrnn';
            [fname, params] = mrbm_filename(params);
            return;
        end
        params.sp_damp = 0.9;
        fname = sprintf('%s_%s_x_%d_z_%d_h_%d_eps_%g_%g_%s_target_%g_reg_%g_l2r_%g_bs_%d_init_%g_up_%d_%d_down_%d_mom_%d_pcd_%d_%d_neg_%d_nmf_%d_px_%g_pz_%g_al_%g', ...
            params.dataset, params.traintype, params.numvx, params.numvz, params.numhid, params.eps, ...
            params.eps_decay, params.sp_type, params.sp_target, params.sp_reg, params.l2reg, ...
            params.batchsize, params.stdinit, params.upfactor_x, params.upfactor_z, ...
            params.downfactor, params.momentum_change, params.usepcd, params.kcd, params.negchain, ...
            params.nmf, params.px, params.pz, params.alpha);
        
    otherwise
        error('undefined model types!');
end

params.fname = fname;
disp(fname);

return;
