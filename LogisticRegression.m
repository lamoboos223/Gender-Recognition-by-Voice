%% Project - Implement Logistic Regression
% Clear all variables and close all plots
clear all; close all; clc;
%% ******************* Loading Data **********************
lambda_values = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];
threshold = 0.5;

%Trainig data
TrainData = load('TrainData.csv');
disp('The dataset was loaded sucessfully!');
TrainX = TrainData(:,1:end-1);% features
TrainY = TrainData(:,end);% class labels

% plot the class Histogram.
figure(1);
hist(TrainY);

%Tsting data
TestData = load('TestData.csv');
disp('The test dataset was loaded sucessfully!');
TestX = TestData(:,1:end-1);% features
TestY = TestData(:,end);% class labels
%% Testing + No normalization
%Training part
disp('without normalization.');
[theta, lambda] = TrainLRModel(TrainX, TrainY, lambda_values);
fprintf('The selected Lambda value is:%f\n', lambda); % = 0.00
disp('Theta values are:');
disp(theta);
y_predicted1 = PredictClass(TestX, theta, threshold);
TestPerformance(TestY, y_predicted1);

%% Testing + normalization

% Training part
TrainXNorm = normalizeFeatures(TrainX); % normalized features.
TestXNorm = normalizeFeatures(TestX); % normalized features.
[theta, lambda] = TrainLRModel(TrainXNorm, TrainY, lambda_values);
fprintf('The selected Lambda value is:%f\n', lambda); % 0.001
disp('Theta values are:');
disp(theta);
y_predicted2 = PredictClass(TestXNorm, theta, threshold);
TestPerformance(TestY, y_predicted2);

%% Final Code
disp("Done.");