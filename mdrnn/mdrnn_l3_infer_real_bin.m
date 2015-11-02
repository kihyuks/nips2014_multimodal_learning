function [topprob, pxprob, pzprob, hxprob, hzprob, data_x, data_z] = mdrnn_l3_infer_real_bin(data_x, data_z, weights, params, nmf, verbose)

if ~exist('nmf', 'var'),
    nmf = 100;
end
if ~exist('verbose', 'var'),
    verbose = 0;
end

batchsize = max(size(data_x, 2), size(data_z, 2));

data_x = double(data_x);
data_z = double(data_z);

vxhx_bu = params.upfactor_vx*weights.vxhx;
vzhz_bu = params.upfactor_vz*weights.vzhz;
vxhx_td = params.downfactor_vx*weights.vxhx;
vzhz_td = params.downfactor_vz*weights.vzhz;
hxpx_bu = params.upfactor_hx*weights.hxpx;
hxpx_td = params.downfactor_hx*weights.hxpx;
hzpz_bu = params.upfactor_hz*weights.hzpz;
hzpz_td = params.downfactor_hz*weights.hzpz;
pxtop_bu = params.upfactor_x*weights.pxtop;
pxtop_td = params.downfactor*weights.pxtop;
pztop_bu = params.upfactor_z*weights.pztop;
pztop_td = params.downfactor*weights.pztop;


tbiasmat = repmat(weights.topbias, [1 batchsize]);
pxbiasmat = repmat(weights.pxbias, [1 batchsize]);
pzbiasmat = repmat(weights.pzbias, [1 batchsize]);
hxbiasmat = repmat(weights.hxbias, [1 batchsize]);
hzbiasmat = repmat(weights.hzbias, [1 batchsize]);
xbiasmat = repmat(weights.vxbias, [1 batchsize]);
zbiasmat = repmat(weights.vzbias, [1 batchsize]);


if isempty(data_x),
    
    error('NOT IMPLEMENTED');
    
elseif isempty(data_z),
    
    hxprob = sigmoid(vxhx_bu'*data_x + hxbiasmat);
    pxprob = sigmoid(hxpx_bu'*hxprob + pxbiasmat);
    wpx = pxtop_bu'*pxprob;
    
    % hidden
    upfactor_x_init = (params.upfactor_x + params.upfactor_z)/params.upfactor_x;
    topprob = sigmoid(upfactor_x_init*wpx + tbiasmat);
    topprob_old = topprob;
    
    for i = 1:nmf,
        % pen (z)
        pzprob = sigmoid(pztop_td*topprob + pzbiasmat);
        
        % hidden (z)
        hzprob = sigmoid(hzpz_td*pzprob + hzbiasmat);
        
        % visible (z)
        data_z = sigmoid(vzhz_td*hzprob + zbiasmat);
        
        % hidden (z)
        hzprob = sigmoid(vzhz_bu'*data_z + hzbiasmat);
        
        % pen
        pzprob = sigmoid(hzpz_bu'*hzprob + pzbiasmat);
        
        % hidden
        topprob = sigmoid(wpx + pztop_bu'*pzprob + tbiasmat);
        
        diff = mean(mean(abs(topprob - topprob_old)));
        if verbose,
            fprintf('[%d/%d] diff = %g\n', i, nmf, diff);
        end
        if diff < 1e-6,
            break;
        else
            topprob_old = topprob;
        end
    end
    
    % pen (z)
    pzprob = sigmoid(pztop_td*topprob + pzbiasmat);
    
    % hidden (z)
    hzprob = sigmoid(hzpz_td*pzprob + hzbiasmat);
    
    % visible (z)
    data_z = sigmoid(vzhz_td*hzprob + zbiasmat);
else
    
    hxprob = sigmoid(vxhx_bu'*data_x + hxbiasmat);
    pxprob = sigmoid(hxpx_bu'*hxprob + pxbiasmat);
    wpx = pxtop_bu'*pxprob;
    
    hzprob = sigmoid(vzhz_bu'*data_z + hzbiasmat);
    pzprob = sigmoid(hzpz_bu'*hzprob + pzbiasmat);
    wpz = pztop_bu'*pzprob;
    
    % hidden
    topprob = sigmoid(wpx + wpz + tbiasmat);
end

return;
