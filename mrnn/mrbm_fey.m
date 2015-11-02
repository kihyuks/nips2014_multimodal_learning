function fey = mrbm_fey(weights, vx, vz, hid, params)

vxhid = weights.vxhid;
vzhid = weights.vzhid;
vxbias = weights.vxbias;
vzbias = weights.vzbias;
hidbias = weights.hidbias;

% visbias
fey = -vxbias'*vx - vzbias'*vz;

if exist('hid', 'var') && ~isempty(hid),
    % compute energy instead of free-energy
    % cond - hid, vx - hid, hidbias
    fey = fey - params.downfactor*sum(vx.*(vxhid*hid), 1);
    fey = fey - params.downfactor*sum(vz.*(vzhid*hid), 1);
    fey = fey - params.downfactor*hidbias'*hid;
else
    % compute free energy
    % cond, vx to hid
    fey = fey - params.downfactor*sum(logexp(bsxfun(@plus, vxhid'*vx + vzhid'*vz, hidbias)), 1);
end

return;