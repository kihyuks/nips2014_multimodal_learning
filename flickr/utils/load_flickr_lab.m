% -------------------------------------------------------------------------
% load labelled image + text data from Flickr database
%
%   xlab    : features from labelled data, both image features and
%             text features are concatenated, D x N (example)
%             e.g) xlab_img = xlab(1:numdim_img, :);
%                  xlab_text = xlab(numdim_img+1:end, :);
%   ylab    : L x N, binary matrix
%   folds   : N x 5, 1 for training, 2 for validation, 3 for testing
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------

function [xlab, ylab, folds, numdim_img, numdim_text, numlabel] = load_flickr_lab

startup;
datadir = [dataroot '/flickr/'];

nfolds = 5;
tr_id = cell(nfolds, 1);
val_id = cell(nfolds, 1);
ts_id = cell(nfolds, 1);

% split
splitdir = [datadir 'splits/'];

for fold_id = 1:nfolds,
    tr_id{fold_id} = load([splitdir 'train_indices_' num2str(fold_id) '.mat'], 'idx');
    val_id{fold_id} = load([splitdir 'valid_indices_' num2str(fold_id) '.mat'], 'idx');
    ts_id{fold_id} = load([splitdir 'test_indices_' num2str(fold_id) '.mat'], 'idx');
end

tot_ex = sum(length(tr_id{1}.idx) + length(val_id{1}.idx) + length(ts_id{1}.idx));

folds = zeros(tot_ex, nfolds);
for fold_id = 1:nfolds,
    folds(double(tr_id{fold_id}.idx+1), fold_id) = 1;
    folds(double(val_id{fold_id}.idx+1), fold_id) = 2;
    folds(double(ts_id{fold_id}.idx+1), fold_id) = 3;
end
clear tr_id val_id ts_id;


% label (multi-label)
load([datadir 'labels.mat'], 'labels');
ylab = labels';
numlabel = size(ylab, 1);

clear labels;


% image
imgdir = [datadir 'image/labelled'];
fnames = dir(imgdir);
Fimg = [];
for i = 1:length(fnames),
    if strcmp(fnames(i).name, '.') || strcmp(fnames(i).name, '..'),
        continue;
    end
    [~, rem] = strtok(fnames(i).name, '.');
    if strcmp(rem, '.mat'),
        load(sprintf('%s/%s', imgdir, fnames(i).name), 'feat');
        Fimg = [Fimg feat'];
    end
end
numdim_img = size(Fimg, 1);

clear feat;


% text
fname = sprintf('%s/text/text_all_2000_labelled.mat', datadir);
Ftext = read_sparse(fname);
Ftext = Ftext';
Ftext(Ftext > 0) = 1; % make sure all entries are 1 (not word count, but word existence)
Ftext = full(Ftext);
numdim_text = size(Ftext, 1);

% concatenate
xlab = [Fimg ; Ftext];

clear Fimg Ftext;

return;


function mat = read_sparse(filename)

load(filename, 'indptr', 'indices', 'data', 'shape');

mat = zeros(shape(1), shape(2));

for i = 1:length(indptr)-1,
    colidx = indices(indptr(i)+1:indptr(i+1)) + 1;
    data_for_row_i = data(indptr(i)+1:indptr(i+1));
    for j = 1:length(colidx),
        mat(i, colidx(j)) = mat(i, colidx(j)) + data_for_row_i(j);
    end
end

mat = sparse(mat);

return;