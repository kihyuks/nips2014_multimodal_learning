function [diff, numgrad, testgrad] = GradCheck(J, theta)
%%% J: function, [cost, grad] = J(theta)
checkNumericalGradient();

[~, testgrad] = J(theta);
numgrad = computeNumericalGradient(@(x) J(x), theta);

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-testgrad)/norm(numgrad+testgrad);
disp(diff);

return;