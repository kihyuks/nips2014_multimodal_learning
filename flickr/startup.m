addpath(genpath('../rbmtoolbox/'));
addpath(genpath('../utils/'));
addpath(genpath('utils/'));
addpath ../mrnn/;
addpath ../mdrnn/;

logdir = [pwd '/log/'];
if ~exist(logdir, 'dir'),
    mkdir(logdir);
end

savedir = [pwd '/results/'];
if ~exist(savedir, 'dir'),
    mkdir(savedir);
end

dataroot = 'data';