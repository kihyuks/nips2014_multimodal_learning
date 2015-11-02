function [Ftext, numdim_text] = load_flickr_unlab_text(numdata)

if ~exist('numdata', 'var'),
    numdata = inf;
end

startup;
datadir = [dataroot '/flickr'];

% text
fname = [datadir '/text/text_all_2000_unlabelled.mat'];
Ftext = read_sparse(fname);
Ftext = Ftext';
Ftext(Ftext > 0) = 1; % make sure all entries are 1 (not word count, but word existence)

% take only first min(numdata, size(Ftext, 2)) examples
Ftext = Ftext(:, 1:min(numdata, size(Ftext, 2)));
Ftext = full(Ftext);
numdim_text = size(Ftext, 1);

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