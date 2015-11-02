function [h, data_x, data_z] = mrbm_infer_bin_bin(data_x, data_z, weights, params, nmf)

if ~exist('nmf', 'var'),
    nmf = 100;
end

batchsize = max(size(data_x, 2), size(data_z, 2));

vxhid_bu = params.upfactor_x*weights.vxhid;
vxhid_td = params.downfactor*weights.vxhid;
vzhid_bu = params.upfactor_z*weights.vzhid;
vzhid_td = params.downfactor*weights.vzhid;

hbiasmat = repmat(weights.hidbias, [1 batchsize]);
xbiasmat = repmat(weights.vxbias, [1 batchsize]);
zbiasmat = repmat(weights.vzbias, [1 batchsize]);

if isempty(data_x),
    wz = vzhid_bu'*data_z;
    
    % hidden
    upfactor_z_init = (params.upfactor_x + params.upfactor_z)/params.upfactor_z;
    h = sigmoid(upfactor_z_init*wz + hbiasmat);
    h_old = h;
    for i = 1:nmf,
        % vis (x)
        data_x = sigmoid(vxhid_td*h + xbiasmat);
        
        % hidden
        h = sigmoid(vxhid_bu'*data_x + wz + hbiasmat);
        
        diff = mean(mean(abs(h - h_old), 2), 1);
        fprintf('[%d/%d] diff = %g\n', i, nmf, diff);
        if diff < 1e-5,
            break;
        else
            h_old = h;
        end
    end
    data_x = sigmoid(vxhid_td*h + xbiasmat);
    
elseif isempty(data_z),
    wx = vxhid_bu'*data_x;
    
    % hidden
    upfactor_x_init = (params.upfactor_x + params.upfactor_z)/params.upfactor_x;
    h = sigmoid(upfactor_x_init*wx + hbiasmat);
    h_old = h;
    for i = 1:nmf,
        % vis (z)
        data_z = sigmoid(vzhid_td*h + zbiasmat);
        
        % hidden
        h = sigmoid(wx + vzhid_bu'*data_z + hbiasmat);
        
        diff = mean(mean(abs(h - h_old), 2), 1);
        fprintf('[%d/%d] diff = %g\n', i, nmf, diff);
        if diff < 1e-5,
            break;
        else
            h_old = h;
        end
    end
    data_z = sigmoid(vzhid_td*h + zbiasmat);
    
else
    % inference
    h = sigmoid(vxhid_bu'*data_x + vzhid_bu'*data_z + hbiasmat);
    
    data_x = sigmoid(vxhid_td*h + xbiasmat);
    data_z = sigmoid(vzhid_td*h + zbiasmat);
end


return;