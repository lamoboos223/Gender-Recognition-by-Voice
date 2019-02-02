function [theta, lambda] = TrainLRModel(x, y, lambda_values)

validationErrors = zeros(10, 1); % this is same as validationError just different name
trainingErrors = zeros(10, 1);
for L=1: length(lambda_values)
    
   cv = cvpartition(y, 'k', 10);
   
%  this is used for ploting the data.
   trainErr = zeros(cv.NumObservations,1);
   validErr = zeros(cv.NumTestSets,1);
   
   theta = learnLRTheta(x, y, lambda_values(L));

%    fprintf('labda= # %d\n', lambda_values(L));
   
    for i=1: cv.NumTestSets


        %Training parts.
        TrainY = y(cv.training(i),:);
        TrainX = x(cv.training(i),:);
        %Validate parts.
        ValidY = y(cv.test(i), :);
        ValidX = x(cv.test(i), :);
        
%         this is just to check if cvpartition worked correctly, because we tried it on smaller dataset
%         fprintf('i= # %d\n', i);
%         disp('the training example is:');
%         disp(TrainX(:, 1));
%         disp('the validation example is:');
%         disp(ValidX(:, 1));
%         disp('======');
        
        TrainCost = calculateCost(TrainX, TrainY, theta, 0); % send lambda with 0 --> no Req
        trainErr(i) = TrainCost;

        ValidationCost = calculateCost(ValidX, ValidY, theta, 0); % send lambda with 0 --> no Req
        validErr(i) = ValidationCost;

    end

% taking the avg of errors in order to plot it later
trainingErrors(L) = mean(trainErr); 
validationErrors(L) = mean(validErr); 
% trainingErrors(L) = sum(trainErr)/cv.TrainSize(i);
% validationErrors(L) = sum(validErr)/cv.TestSize(i);

end

% ploting the error curves.
figure();
x_axis = lambda_values;
y_axis1 = validationErrors;
y_axis2 = trainingErrors;
% plot(x_axis, y_axis1, x_axis,y_axis2);
plot(x_axis,y_axis1,'LineWidth',2)
hold on
plot(x_axis,y_axis2,'LineWidth',2)
hold off
xlabel('Lambda Values') 
ylabel('Training& Validatoion cost functions.') 
legend({'Validation','Training'},'Location','southwest')


[M, I] = min(validationErrors);
fprintf('The lowest error is: %f\n', M);
lambda = lambda_values(I); % return the lambda with minimum validation Error
theta  = learnLRTheta(x, y, lambda);
% disp(trainingErrors);
% disp(validationErrors);
end



