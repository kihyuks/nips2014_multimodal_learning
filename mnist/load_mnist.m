% -------------------------------------------------------------------------
% load mnist handwritten digits from the web
% xtrain, xval, xtest: 784 x # samples
% ytrain, yval, ytest: 1 x # samples
% 
% written by Kihyuk Sohn, 2015.06.17
% -------------------------------------------------------------------------

function [xtrain, ytrain, xval, yval, xtest, ytest, dim] = load_mnist

if exist('data/mnist_train.mat', 'file'),
    load data/mnist_train.mat digits labels;
    xtrain = digits;
    ytrain = labels;
    clear digits labels;
else
    [xtrain, ytrain] = load_mnist_from_web('train');
end

xval = xtrain(:, 50001:end);
yval = ytrain(50001:end);
xtrain = xtrain(:, 1:50000);
ytrain = ytrain(1:50000);


if exist('data/mnist_test.mat', 'file'),
    load data/mnist_test.mat digits labels;
    xtest = digits;
    ytest = labels;
    clear digits labels;
else
    [xtest, ytest] = load_mnist_from_web('test');
end

dim = [28 28 1];

return;


function [digits, labels] = load_mnist_from_web(mode)

if strcmp(mode, 'train'),
    system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz');
    system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz');
    fname_image = 'train-images-idx3-ubyte';
    fname_label = 'train-labels-idx1-ubyte';
    system(['gunzip ' fname_image '.gz']);
    system(['gunzip ' fname_label '.gz']);
    nsample = 60000;
elseif strcmp(mode, 'test'),
    system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz');
    system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz');
    fname_image = 't10k-images-idx3-ubyte';
    fname_label = 't10k-labels-idx1-ubyte';
    system(['gunzip ' fname_image '.gz']);
    system(['gunzip ' fname_label '.gz']);
    nsample = 10000;
end


% digit images
f = fopen(fname_image, 'r');
fread(f, 4, 'int32');
digits = fread(f, 28^2*nsample, 'uchar');
digits = reshape(digits, 28, 28, nsample);
digits = permute(digits, [2 1 3]);
digits = reshape(digits, 28^2, nsample);
digits = digits./255.0; % values in [0, 1]
fclose(f);

% label
g = fopen(fname_label, 'r');
fread(g, 2, 'int32');
labels = fread(g, nsample, 'uchar');
labels = labels + 1; % values from 1 to 10
fclose(g);

if strcmp(mode, 'train'),
    rng('default');
    idx = randperm(length(labels));
    digits = digits(:, idx);
    labels = labels(idx);
end

if ~exist('data', 'dir'),
    mkdir('data');
end

switch mode,
    case 'train',
        save('data/mnist_train.mat', 'digits', 'labels');
    case 'test',
        save('data/mnist_test.mat', 'digits', 'labels');
end

system(['rm ' fname_image]);
system(['rm ' fname_label]);

return;

