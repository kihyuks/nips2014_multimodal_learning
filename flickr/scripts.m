if ~exist('optgpu', 'var'),
    error('define optgpu (0/1) and initialize with gpuDevice')
end

map_test = [];
fname = {};


% -----------------------------
% first layer pretraining
% -----------------------------

% [~, ~, map_test(end+1)] = flickr_img_l1(1,1:5,optgpu,1024,5,0.001,1,0.2,1e-5,1,300,100,0.005,2,1);
% fname{end+1} = 'flickr_img_l1';
% [~, ~, map_test(end+1)] = flickr_text_l1(1,1:5,optgpu,5,1024,1,0.1,0.01,0.2,0,1,200,100,0.005,2,1);
% fname{end+1} = 'flickr_text_l1';


% -----------------------------
% second layer pretraining
% -----------------------------

% [~, ~, map_test(end+1)] = flickr_img_l2(1,1:5,optgpu,...
%     1024,5,0.001,1,0.2,1e-5,1,300,100,0.005,2,1,1024,1,0.1,1,0.2,1e-5,1,200,200,0.005,2,2);
% fname{end+1} = 'flickr_img_l2';
% [~, ~, map_test(end+1)] = flickr_text_l2(1,1:5,optgpu,5,...
%     1024,1,0.1,0.01,0.2,0,1,200,100,0.005,2,1,1024,1,0.03,1,0.2,1e-5,1,100,200,0.005,2,2);
% fname{end+1} = 'flickr_text_l2';


% -----------------------------
% pretrain top layer
% -----------------------------

% MRBM with VI using MP training (proposed)
[~, map_test(end+1), ~, map_test(end+1)] = ...
    flickr_both_l3(1,1:5,optgpu,2048,0.01,0.01,0.01,0.2,1e-5,10,200,200,0,0);
fname{end+1} = 'flickr_both_l3_top_mult';
fname{end+1} = 'flickr_both_l3_top_uni';


% -----------------------------
% joint training
% -----------------------------

% MRBM with VI using MP training (proposed)
[~, map_test(end+1), ~, map_test(end+1)] = ...
    flickr_both_l3(1,1:5,optgpu,2048,0.01,0.01,0.01,0.2,1e-5,10,200,200,0,0,1,0.01,0.01,0.01,1e-5,10,200,0,0);
fname{end+1} = 'flickr_both_l3_joint_mult';
fname{end+1} = 'flickr_both_l3_joint_uni';


% -----------------------------
% fine-tuning
% -----------------------------

% MRBM with VI using MP training (proposed; x -> z)
[~, map_test(end+1), ~, map_test(end+1)] = ...
    flickr_both_l3(1,1:5,optgpu,2048,0.01,0.01,0.01,0.2,1e-5,10,200,200,0,0,1,0.01,0.01,0.01,1e-5,10,200,0,0,1,0.03,0.01,0.01,1e-5,10,200,1,0.1);
fname{end+1} = 'flickr_both_l3_finetune_mult';
fname{end+1} = 'flickr_both_l3_finetune_uni';


% print result
fprintf('\n\n\n');
fprintf('=======================================================================\n');
fprintf('recognition performance (test set)\n');
fprintf('\n\n');
for i = 1:length(fname),
    fprintf('%s:\ttest = %.4f\n', fname{i}, map_test(i));
end
fprintf('\n\n');
fprintf('=======================================================================\n');
fprintf('\n\n\n');

