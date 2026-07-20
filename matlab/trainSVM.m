clc;
clear;
close all;

%% Read Dataset

data = readtable('/MATLAB Drive/ECG_Arrhythmia_Project/data/heartbeat_dataset.csv');
disp('Dataset Loaded Successfully');
disp(head(data));

%% Features

X = data{:, {'RR_Previous','RR_Next','Heart_Rate'}};

%% Labels

Y = categorical(data.Label);

disp(categories(Y));

%% ===============================
% Prepare Features and Labels
% ================================

X = data{:, {'RR_Previous','RR_Next','Heart_Rate'}};

Y = categorical(data.Label);

%% ===============================
% Split Data (80% Training, 20% Testing)
% ================================

cv = cvpartition(Y,'HoldOut',0.2);

XTrain = X(training(cv),:);
YTrain = Y(training(cv));

XTest = X(test(cv),:);
YTest = Y(test(cv));

disp('Training Data Size:')
disp(size(XTrain))

disp('Testing Data Size:')
disp(size(XTest))

%% ===============================
% Train SVM
% ================================

SVMModel = fitcecoc(XTrain,YTrain);

disp('SVM Model Trained Successfully!')

%% ===============================
% Predict Test Data
% ================================

YPred = predict(SVMModel,XTest);

disp('Prediction Completed!')

%% ===============================
% Accuracy
% ================================

accuracy = sum(YPred == YTest) / numel(YTest);

fprintf('\nAccuracy = %.2f%%\n', accuracy*100);
%% ===============================
% Confusion Matrix
% ===============================

figure;
confusionchart(YTest, YPred);

title('Confusion Matrix');
%% ===============================
% Performance Metrics
% ===============================

cm = confusionmat(YTest, YPred);

disp('Confusion Matrix');
disp(cm);

% Overall Accuracy
accuracy = sum(diag(cm)) / sum(cm(:));

fprintf('\nOverall Accuracy = %.2f%%\n', accuracy*100);

%% Save Trained Model

save('../results/SVM_Model.mat', 'SVMModel');

disp('Model saved successfully!');

figure
