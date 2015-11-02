% -------------------------------------------------------------------------
% load unlabelled image + text data from Flickr database
%
%   doc_length  : minimum length of tags to be used for training
%   numdata     : number of data per batch (total 100 batches)
%
%   xunlab      : features from unlabelled data, both image features and
%                 text features are concatenated, D x N (example)
%                 e.g) xunlab_img = xunlab(1:numdim_img, :);
%                      xunlab_text = xunlab(numdim_img+1:end, :);
%
%   written by Kihyuk Sohn
% -------------------------------------------------------------------------

function [xunlab, numdim_img, numdim_text] = load_flickr_unlab(doc_length, numdata)

if ~exist('doc_length', 'var'),
    doc_length = 2;
end
if ~exist('numdata', 'var'),
    numdata = 10000; % # data per batch
end

startup;
rng('default');


% text data
fprintf('load text data ..');
[x_text, numdim_text] = load_flickr_unlab_text;
x_text = single(x_text);
fprintf('done\n');

% find data with tag length >= doc_length
idx = sum(x_text, 1) >= doc_length;


% image
datadir = [dataroot '/flickr/image/unlabelled/'];

datalist = dir(datadir);
datalist(1:2) = [];
datafile = cell(length(datalist), 1);
for i = 1:length(datalist),
    datafile{i} = [datadir datalist(i).name];
end

[~, ~, ~, numdim_img, ~, ~] = load_flickr_lab;
xunlab = single(zeros(numdim_img + numdim_text, numdata*length(datafile)));

fprintf('load image data ');

k = 0;
unlab_id = 0;
for i = 1:length(datafile),
    if ~mod(i, 10),
        fprintf('.');
    end
    xt = load_datafile(datafile{i}, numdim_img);
    xt = single(xt);
    
    xt_all = [xt ; x_text(:, k+1:k+size(xt, 2))];
    xt_idx = idx(k+1:k+size(xt, 2));
    
    % remove examples with small tags
    xt_all = xt_all(:, xt_idx);
    
    sample_idx = randsample(size(xt_all, 2), min(size(xt_all, 2), numdata));
    xunlab(:, unlab_id+1:unlab_id+length(sample_idx)) = xt_all(:, sample_idx);
    
    unlab_id = unlab_id + length(sample_idx);
    k = k + size(xt, 2);
end
clear xt xt_all x_text;

xunlab = single(xunlab);
xunlab = xunlab(:, 1:unlab_id);
fprintf('done!\n');

return;


% -------------------------------------------------------------------------
%                                                           Load datafile
% -------------------------------------------------------------------------

function x = load_datafile(datafile, numdim)

x = load(datafile);
fname = fieldnames(x);
x = getfield(x, fname{1});
if size(x, 1) ~= numdim,
    x = x';
end

return
