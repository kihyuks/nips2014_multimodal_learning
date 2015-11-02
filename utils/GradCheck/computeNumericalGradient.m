function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));
epsilon = 1e-4;

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

ln = ceil(length(theta)/100);
for i = 1:length(theta),
    if ~mod(i,ln)
        fprintf('[%d/100]', round(i/ln));
    end
    if ~mod(i,10*ln),
        fprintf('\n');
    end
    theta_pos = theta;
    theta_neg = theta;
    theta_pos(i) = theta_pos(i) + epsilon;
    theta_neg(i) = theta_neg(i) - epsilon;
    numgrad(i) = (J(theta_pos)-J(theta_neg))/2/epsilon;
end
fprintf('\n\n');

return;
