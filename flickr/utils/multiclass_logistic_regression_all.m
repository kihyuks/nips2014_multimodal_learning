function [map_train, map_val, map_test, l2reg, w, b] = multiclass_logistic_regression_all(l2reg_list, xtrain, ytrain, xval, yval, xtest, ytest)

xtrain = double(xtrain);
ytrain = double(ytrain);
xval = double(xval);
yval = double(yval);
xtest = double(xtest);
ytest = double(ytest);

if isempty(l2reg_list),
    l2reg_list = [0 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10];
end

numlabel = size(ytrain, 1);

map_val_best = 0;

% validation
for i = 1:length(l2reg_list),
    l2reg = l2reg_list(i);
    [w, b] = multiclass_logistic_regression(xtrain, ytrain, l2reg);
    
    score_val = bsxfun(@plus, w'*xval, b);
    map_val = zeros(numlabel, 1);
    for l = 1:numlabel,
        [~, ~, ap] = compute_ap(score_val(l, :), yval(l, :));
        map_val(l) = ap;
    end
    map_val = mean(map_val);
    if length(l2reg_list) > 1,
        fprintf('mAP (val) = %g, l2reg = %g\n', map_val, l2reg);
    end
    if map_val > map_val_best,
        l2reg_best = l2reg;
        map_val_best = map_val;
        wbest = w;
        bbest = b;
    end
end

l2reg = l2reg_best;
w = wbest;
b = bbest;

% train
score = bsxfun(@plus, w'*xtrain, b);
map = zeros(numlabel, 1);
for l = 1:numlabel,
    [~, ~, ap] = compute_ap(score(l, :), ytrain(l, :));
    map(l) = ap;
end
map_train = mean(map);

% validation
score = bsxfun(@plus, w'*xval, b);
map = zeros(numlabel, 1);
for l = 1:numlabel,
    [~, ~, ap] = compute_ap(score(l, :), yval(l, :));
    map(l) = ap;
end
map_val = mean(map);

% test
score = bsxfun(@plus, w'*xtest, b);
map = zeros(numlabel, 1);
for l = 1:numlabel,
    [~, ~, ap] = compute_ap(score(l, :), ytest(l, :));
    map(l) = ap;
end
map_test = mean(map);

fprintf('CV mAP:  train = %g, val = %g, test = %g, l2reg = %g\n', map_train, map_val, map_test, l2reg);

return;


