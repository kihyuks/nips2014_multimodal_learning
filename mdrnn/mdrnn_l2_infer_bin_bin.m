function [penprob, hxprob, hzprob, data_x, data_z] = mdrnn_l2_infer_bin_bin(data_x, data_z, weights, params, nmf, verbose)

if ~exist('nmf', 'var'),
    nmf = 100;
end
if ~exist('verbose', 'var'),
    verbose = 1;
end

batchsize = max(size(data_x, 2), size(data_z, 2));

data_x = double(data_x);
data_z = double(data_z);

vxhx_bu = double(params.upfactor_vx*weights.vxhx);
vxhx_td = double(params.downfactor_vx*weights.vxhx);
vzhz_bu = double(params.upfactor_vz*weights.vzhz);
vzhz_td = double(params.downfactor_vz*weights.vzhz);
hxpen_bu = double(params.upfactor_x*weights.hxpen);
hxpen_td = double(params.downfactor*weights.hxpen);
hzpen_bu = double(params.upfactor_z*weights.hzpen);
hzpen_td = double(params.downfactor*weights.hzpen);

pbiasmat = double(repmat(weights.penbias, [1 batchsize]));
hxbiasmat = double(repmat(weights.hxbias, [1 batchsize]));
hzbiasmat = double(repmat(weights.hzbias, [1 batchsize]));
xbiasmat = double(repmat(weights.vxbias, [1 batchsize]));
zbiasmat = double(repmat(weights.vzbias, [1 batchsize]));

if isempty(data_x),
    hzprob = sigmoid(vzhz_bu'*data_z + hzbiasmat);
    whz = hzpen_bu'*hzprob;
    
    % hidden
    upfactor_z_init = (params.upfactor_z + params.upfactor_x)/params.upfactor_z;
    penprob = sigmoid(upfactor_z_init*whz + pbiasmat);
    penprob_old = penprob;
    
    for i = 1:nmf,
        % hidden (x)
        hxprob = sigmoid(hxpen_td*penprob + hxbiasmat);
        
        % visible (x)
        data_x = sigmoid(vxhx_td*hxprob + xbiasmat);
        
        % hidden (x)
        hxprob = sigmoid(vxhx_bu'*data_x + hxbiasmat);
        
        % hidden
        penprob = sigmoid(whz + hxpen_bu'*hxprob + pbiasmat);
        
        diff = mean(mean(abs(penprob - penprob_old), 2), 1);
        if verbose,
            fprintf('[%d/%d] diff = %g\n', i, nmf, diff);
        end
        if diff < 1e-5,
            break;
        else
            penprob_old = penprob;
        end
    end
    % hidden (x)
    hxprob = sigmoid(hxpen_td*penprob + hxbiasmat);
    
    % visible (x)
    data_x = sigmoid(vxhx_td*hxprob + xbiasmat);
    
elseif isempty(data_z),
    
    hxprob = sigmoid(vxhx_bu'*data_x + hxbiasmat);
    whx = hxpen_bu'*hxprob;
    
    
    % hidden
    upfactor_x_init = (params.upfactor_x + params.upfactor_z)/params.upfactor_x;
    penprob = sigmoid(upfactor_x_init*whx + pbiasmat);
    penprob_old = penprob;
    
    for i = 1:nmf,
        % hidden (z)
        hzprob = sigmoid(hzpen_td*penprob + hzbiasmat);
        
        % visible (z)
        data_z = sigmoid(vzhz_td*hzprob + zbiasmat);
        
        % hidden (z)
        hzprob = sigmoid(vzhz_bu'*data_z + hzbiasmat);
        
        % hidden
        penprob = sigmoid(whx + hzpen_bu'*hzprob + pbiasmat);
        
        diff = mean(mean(abs(penprob - penprob_old), 2), 1);
        if verbose,
            fprintf('[%d/%d] diff = %g\n', i, nmf, diff);
        end
        if diff < 1e-5,
            break;
        else
            penprob_old = penprob;
        end
    end
    % hidden (z)
    hzprob = sigmoid(hzpen_td*penprob + hzbiasmat);
    
    % visible (z)
    data_z = sigmoid(vzhz_td*hzprob + zbiasmat);
    
else
    hxprob = sigmoid(vxhx_bu'*data_x + hxbiasmat);
    whx = hxpen_bu'*hxprob;
    
    hzprob = sigmoid(vzhz_bu'*data_z + hzbiasmat);
    whz = hzpen_bu'*hzprob;
    
    penprob = sigmoid(whx + whz + pbiasmat);
end


return;