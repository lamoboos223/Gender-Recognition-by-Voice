%% ********* Learn Theta Values *********

function theta = learnLRTheta(X, y, lambda)
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
 
%  This function will return the learned theta values  
theta= ...
    fminunc(@(t)(calculateCost(X, y, t, lambda)),zeros(size(X,2),1), options);
 
end