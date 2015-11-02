% ----------------------------------
% Fill in missing parameters
% ----------------------------------

function params = fillin_params(params)

if ~isfield(params, 'typein'),      params.typein = 'real';             end % 'real', 'binary'
if ~isfield(params, 'typeout'),     params.typeout = 'binary';          end % 'binary', 'step'

if ~isfield(params, 'savedir'),     params.savedir = 'results';         end
if ~isfield(params, 'saveiter'),    params.saveiter = 20;               end
if ~isfield(params, 'dataset'),     params.dataset = 'dummy';           end
if ~isfield(params, 'optgpu'),      params.optgpu = 0;                  end
if ~isfield(params, 'maxiter'),     params.maxiter = 100;               end
if ~isfield(params, 'batchsize'),   params.batchsize = 100;             end

% momentum (stochastic gradient descent)
if ~isfield(params, 'momentum_init'),   params.momentum_init = 0.33;    end
if ~isfield(params, 'momentum_final'),  params.momentum_final = 0.5;    end
if ~isfield(params, 'momentum_change'), params.momentum_change = 5;     end

% learning rate
if ~isfield(params, 'eps'),         params.eps = 0.05;                  end
if ~isfield(params, 'eps_decay'),   params.eps_decay = 0.0;             end

% CD training
if ~isfield(params, 'usepcd'),      params.usepcd = 0;                  end
if ~isfield(params, 'kcd'),         params.kcd = 1;                     end
if ~isfield(params, 'negchain'),    params.negchain = params.batchsize; end

% regularizers (l2, l1, sparsity)
if ~isfield(params, 'l2reg'),       params.l2reg = 0.0;                 end
if ~isfield(params, 'l1reg'),       params.l1reg = 0.0;                 end
if ~isfield(params, 'sp_type'),     params.sp_type = 'approx';          end
if ~isfield(params, 'sp_damp'),     params.sp_damp = 0.9;               end
if ~isfield(params, 'sp_reg'),      params.sp_reg = 0;                  end
if ~isfield(params, 'sp_target'),   params.sp_target = 0;               end

% visualizations
if ~isfield(params, 'verbose'),     params.verbose = 1;                 end

% for std learning
if ~isfield(params, 'std_lb'),      params.std_lb = 0.001;              end
if ~isfield(params, 'std_learn'),   params.std_learn = 1;               end
if ~isfield(params, 'std_share'),   params.std_share = 0;               end

if ~isfield(params, 'normalize'),   params.normalize = 0;               end
if ~isfield(params, 'epsnorm'),     params.epsnorm = 0;                 end
if ~isfield(params, 'stdinit'),     params.stdinit = 0.01;              end
if ~isfield(params, 'draw_sample'), params.draw_sample = 1;             end

% stepped-sigmoid units
if ~isfield(params, 'numstep_v'),   params.numstep_v = 1;               end
if ~isfield(params, 'numstep_h'),   params.numstep_h = 1;               end

if ~isfield(params, 'upfactor'),    params.upfactor = 1;                end
if ~isfield(params, 'downfactor'),  params.downfactor = 1;              end

% some sanity check
if ~params.usepcd,
    params.negchain = params.batchsize;
end
if params.sp_reg == 0,
    params.sp_target = 0;
end
if ~exist(params.savedir, 'dir'),
    mkdir(params.savedir);
end
if strcmp(params.typein, 'binary') || strcmp(params.typein, 'real'),
    params.numstep_v = 1;
end
if strcmp(params.typeout, 'binary') || strcmp(params.typeout, 'real'),
    params.numstep_h = 1;
end

return;
