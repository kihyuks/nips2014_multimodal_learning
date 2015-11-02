function [acc_train, acc_val, acc_test, bestC, acc_train_list, acc_val_list, acc_ts_list] = liblinear_wrapper(Clist, xtrain, ytrain, xval, yval, xtest, ytest)

% addpath for liblinear:
% addpath(genpath('liblinear-1.93/matlab/'));

fprintf('start testing... ');
if ~exist('Clist', 'var') || isempty(Clist),
    Clist = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 10, 30, 100, 300, 1000];
end

ytrain = ytrain(:);
yval = yval(:);
if exist('ytest','var'),
    ytest = ytest(:);
end

acc_train_list = zeros(length(Clist), 1);
acc_val_list = zeros(length(Clist), 1);
model_list = cell(length(Clist), 1);
parfor j = 1:length(Clist);
    C = Clist(j);
    opt_svm = sprintf('-s 2 -B 1 -c %g -q', C);
    model = train(ytrain, sparse(double(xtrain)), opt_svm, 'col');
    [~, acc_train] = predict(ytrain, sparse(double(xtrain)), model, '-q','col');
    [~, acc_val] = predict(yval, sparse(double(xval)), model, '-q','col');
    [~, acc_ts] = predict(ytest, sparse(double(xtest)), model, '-q','col');
    acc_train_list(j) = acc_train(1);
    acc_val_list(j) = acc_val(1);
    acc_ts_list(j) = acc_ts(1);
    model_list{j} = model;
end

[acc_val, id] = max(acc_val_list);
bestC = Clist(id);

if ~exist('xtest','var') || isempty(xtest) || ~exist('ytest','var') || isempty(ytest),
    acc_test = [];
    acc_train = acc_train_list(id);
    fprintf('Liblinear after CV: C=%g, trainerr=%g, valerr=%g\n', bestC, 100-acc_train, 100-acc_val);
else
    model = model_list{id};
    
    [~, acc_train] = predict(ytrain, sparse(double(xtrain)), model, '-q','col');
    [~, acc_test] = predict(ytest, sparse(double(xtest)), model, '-q','col');
    fprintf('Liblinear after CV: C=%g, trainerr=%g, valerr=%g, testerr=%g\n', bestC, 100-acc_train(1), 100-acc_val, 100-acc_test(1));
end

acc_train = acc_train(1);
acc_val = acc_val(1);
if ~isempty(acc_test),
    acc_test = acc_test(1);
end

return;
