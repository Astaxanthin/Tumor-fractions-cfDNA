seed = 41; 
rand('seed',seed);

%% training data, test data
file_dir = strcat('../data/',dataset_name,'/');
save_dir = strcat('../results/',dataset_name,'/');
mkdir(save_dir)

train_data_dir = strcat(file_dir,'train_data.mat');
test_data_dir = strcat(file_dir,'test_data.mat');

load(train_data_dir);
load(test_data_dir);

if strcmp(dataset_name,'simulation_dataset')
    load(strcat(file_dir,'train_theta.mat'));
    load(strcat(file_dir,'test_theta.mat'));
    evaluate_deconvolution_flag = true;
elseif strcmp(dataset_name,'real_dataset')
    evaluate_deconvolution_flag = false;
end

%% parameter configuration
global param;

param.p = 0.5;  % 2-p norm
param.lambda = 100;  % coefficient of structural penalty
param.cancer_pattern_num = 3;
param.healthy_pattern_num = 10;
param.convergence_threshold = 1e-2;
param.maximum_iterations = 1000;
param.prior = 'SVM';  %RF, NN, SVM, Equal