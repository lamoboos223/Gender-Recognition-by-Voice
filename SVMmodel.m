% Clear all variables and close all plots
clear all; close all; clc;
%% ******************* Loading Data **********************
%Trainig data
data = load('TrainData.csv');
disp('The dataset was loaded sucessfully!');
TrainX = data(:,1:end-1);% features
TrainY = data(:,end);% class labels

%Tsting data
data = load('TestData.csv');
disp('The test dataset was loaded sucessfully!');
TestX = data(:,1:end-1);% features
TestY = data(:,end);% class labels

%% ******************** SVM Model ***********************
SVMModel=fitcsvm(TrainX,TrainY);%%Train
label=predict(SVMModel,TestX); %%predict new data label(class)
[Accurecy, Recall, Precision, FScore] = TestPerformance(TestY, label);


