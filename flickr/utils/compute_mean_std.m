function [m_global, stds_global] = compute_mean_std(doc_length, imgperbatch)

if ~exist('doc_length', 'var'),
    doc_length = 2;
end
if ~exist('imgperbatch', 'var'),
    imgperbatch = 10000;
end

try
    load(sprintf('data/mean_std_ln_%d_img_%d.mat', doc_length, imgperbatch), 'm_global', 'stds_global');
catch
    [xunlab, numdim_img, numdim_text] = load_flickr_unlab(doc_length, imgperbatch);
    xunlab_img = xunlab(1:numdim_img, :);
    xunlab_text = xunlab(numdim_img+1:end, :);
    clear xunlab;
    
    m_global = mean(xunlab_img, 2);
    stds_global = sqrt(var(xunlab_img, [], 2) + min(var(xunlab_img, [], 2))*1e-3);
    xunlab_img = bsxfun(@rdivide, bsxfun(@minus, xunlab_img, m_global), stds_global);
    save(sprintf('data/mean_std_ln_%d_img_%d.mat', doc_length, imgperbatch), 'm_global', 'stds_global');
end
