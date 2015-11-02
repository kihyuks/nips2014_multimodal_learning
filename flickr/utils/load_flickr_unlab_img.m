function [x, numdim] = load_flickr_unlab_img(numdatafile)

if ~exist('numdatafile', 'var'),
    numdatafile = inf;
end

datadir = 'data/flickr/image/unlabelled/';

datalist = dir(datadir);
datalist(1:2) = [];
datafile = cell(length(datalist), 1);

for i = 1:length(datalist),
    datafile{i} = sprintf('%s%s', datadir, datalist(i).name);
end

x = [];
for i = 1:min(length(datalist), numdatafile),
    load(datafile{i}, 'feat');
    x = [x feat'];
end

numdim = size(x, 1);

return;