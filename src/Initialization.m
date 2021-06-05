global param;
seed = 41;
rand('seed',seed);

%% load training and  test data
file_dir = strcat('../data/',param.dataset_name,'/');

if strcmp(param.dataset_name,'simulation_dataset')
    param.cnv = 30;
    train_data_dir = strcat(file_dir,'CNV',num2str(param.cnv),'/train_data.mat');
    test_data_dir = strcat(file_dir,'CNV',num2str(param.cnv),'/test_data.mat');
    load(train_data_dir);
    load(test_data_dir);
    
    load(strcat(file_dir,'CNV',num2str(param.cnv),'/train_theta.mat'));
    load(strcat(file_dir,'CNV',num2str(param.cnv),'/test_theta.mat'));
    evaluate_deconvolution_flag = true;
    
elseif strcmp(param.dataset_name,'real_dataset')
    data_dir = strcat(file_dir,'real_data.mat');
    load(data_dir);
    [train_data,test_data] = random_data_split(real_data);
    param.top_k = 531; 
    marker_index = select_dscore_markers(train_data);  % select Top-K markers by D-score
    train_data = train_data([marker_index,end],:);
    test_data = test_data([marker_index,end],:);

    evaluate_deconvolution_flag = false;
    param.oversampling_flag = true;
end

%% parameter configuration
param.p = 0.5;  % 2-p norm
param.lambda = 100;  % coefficient of structural penalty
param.cancer_pattern_num = 3;
param.healthy_pattern_num = 10;
param.convergence_threshold = 1e-2;
param.maximum_iterations = 1000;
param.prior = 'RF';  %RF, NN, SVM, Equal
