clc;
clear;
close all;

% Read the ECG feature dataset
data = readtable('../data/ecg_features_dataset.csv');

disp('Dataset Loaded Successfully');
disp(data);

% Display dataset size
disp(' ');
disp(['Number of records: ', num2str(height(data))]);
disp(['Number of features: ', num2str(width(data))]);