addpath ../utils/;
addpath ../mrnn/;

% EDIT: addpath for liblinear library
LIBLINEAR = '~/libdeepnets2/experimental/kihyuks/linearsvm';
addpath(genpath([LIBLINEAR '/liblinear-1.93/matlab/']));

datadir = [pwd '/data'];
if ~exist(datadir, 'dir'),
	mkdir(datadir);
end

logdir = [pwd '/log'];
if ~exist(logdir, 'dir'),
    mkdir(logdir);
end

savedir = [pwd '/results'];
if ~exist(savedir, 'dir'),
    mkdir(savedir);
end
