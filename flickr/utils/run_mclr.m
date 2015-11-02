% -------------------------------------------------------------------------
% run multi-label logistic regression classifier using minFunc
%
%   xlab        : D x N (example)
%   xlab_test   : D x N (example)
%   ylab        : L x N (example), binary matrix
%   folds       : N x nfolds, 1 for train, 2 for validation, 3 for testing
%   fold_list   : fold list to evaluate
%   l2reg_list  : l2reg list to evaluate
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------

function [map_val, map_test, l2reg] = run_mclr(evalmode, xlab, xlab_test, ylab, folds, fold_list, l2reg_list, logdir, prefix, fname)

if ~exist('evalmode', 'var') || isempty(evalmode),
    evalmode = 'val';
end
if ~exist('xlab_test', 'var') || isempty(xlab_test),
    xlab_test = xlab;
end
if ~exist('fold_list', 'var') || isempty(fold_list),
    fold_list = 1;
end
if ~exist('l2reg_list', 'var') || isempty(l2reg_list),
    l2reg_list = 0;
end
if ~exist('logdir', 'var') || isempty(logdir),
    logdir = pwd;
end
if ~exist('prefix', 'var') || isempty(prefix),
    prefix = '';
end
if ~exist('fname', 'var') || isempty(fname),
    fname = '';
end


% run multiclass logistic regression
% validation on first fold
fprintf('start testing fold 1\n');
ap_val_list = zeros(5, 1);
ap_test_list = zeros(5, 1);

fold_id = 1;

switch evalmode,
    case 'val',
        tr_id = folds(:, fold_id) == 1;
        val_id = folds(:, fold_id) == 2;
        ts_id = folds(:, fold_id) == 3;
    case 'test',
        tr_id = logical((folds(:, fold_id) == 1) + (folds(:, fold_id) == 2));
        val_id = folds(:, fold_id) == 3;
        ts_id = folds(:, fold_id) == 3;
end


htrain = xlab(:, tr_id);
ytrain = ylab(:, tr_id);
hval = xlab_test(:, val_id);
yval = ylab(:, val_id);
htest = xlab_test(:, ts_id);
ytest = ylab(:, ts_id);


[~, ap_val, ap_test, l2reg] = multiclass_logistic_regression_all(l2reg_list, htrain, ytrain, hval, yval, htest, ytest);
ap_val_list(fold_id) = ap_val;
ap_test_list(fold_id) = ap_test;

for fold_id = setdiff(fold_list, fold_list(1)),
    fprintf('start testing fold %d\n', fold_id);
    switch evalmode,
        case 'val',
            tr_id = folds(:, fold_id) == 1;
            val_id = folds(:, fold_id) == 2;
            ts_id = folds(:, fold_id) == 3;
        case 'test',
            tr_id = logical((folds(:, fold_id) == 1) + (folds(:, fold_id) == 2));
            val_id = folds(:, fold_id) == 3;
            ts_id = folds(:, fold_id) == 3;
    end
    
    htrain = xlab(:, tr_id);
    ytrain = ylab(:, tr_id);
    hval = xlab_test(:, val_id);
    yval = ylab(:, val_id);
    htest = xlab_test(:, ts_id);
    ytest = ylab(:, ts_id);
    
    [~, ap_val, ap_test] = multiclass_logistic_regression_all(l2reg, htrain, ytrain, hval, yval, htest, ytest);
    ap_val_list(fold_id) = ap_val;
    ap_test_list(fold_id) = ap_test;
end


% log
if length(fold_list) == 5,
    map_val = mean(ap_val_list, 1);
    std_val = std(ap_val_list);
    map_test = mean(ap_test_list, 1);
    std_test = std(ap_test_list);
    
    fprintf('map: val = %g (std = %g), test = %g (std = %g)\n', map_val, std_val, map_test, std_test);
    if ~isempty(logdir) && ~isempty(prefix),
        fid = fopen(sprintf('%s/%s.txt', logdir, prefix), 'a+');
        fprintf(fid, 'map: val = %g (std = %g), test = %g (std = %g) (%s)\n', map_val, std_val, map_test, std_test, fname);
        fclose(fid);
    end
else
    map_val = mean(ap_val_list(fold_list, :), 1);
    map_test = mean(ap_test_list(fold_list, :), 1);
    
    fprintf('map: val = %g, test = %g\n', map_val, map_test);
    if ~isempty(logdir) && ~isempty(prefix),
        fid = fopen(sprintf('%s/%s_fold%d.txt', logdir, prefix, fold_list), 'a+');
        fprintf(fid, 'map: val = %g, test = %g (%s)\n', map_val, map_test, fname);
        fclose(fid);
    end
end

return;