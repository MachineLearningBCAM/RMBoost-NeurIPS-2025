close all
clear all

%%% Load data

dataset_name = 'diabetes';
f = fullfile('../data',join([dataset_name, '.mat']));
load(f);

X = zscore(X)';%Preprocess data

%%% Hyper-parameters

hyper_param.solver = "linprog";% linprog or Mosek
hyper_param.T = 200; % Number of iterations 

if hyper_param.solver == "Mosek"
    addpath 'Users/.../mosek/10.1/toolbox/r2017a'
end

%%% Fit the model
[model, upper] = fit(X, y, hyper_param);

%%% Prediction
y_pred = predict_boost(model, X_test);

%%% Error
error = sum(y_pred~=y_test)/length(y_test);