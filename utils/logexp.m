function y = logexp(x)
% y = log(1+exp(x)), stable version
%
y = max(0,x) + log(1+exp(-abs(x)));

return;