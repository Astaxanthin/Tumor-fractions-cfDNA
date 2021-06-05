clc;
clear;

%% initialization
global param;
param.dataset_name = 'real_dataset';  % choose a dataset: 'simulation_dataset' and 'real_dataset'
Initialization;  % data loading, parameter setting, etc.

%% sREDBI algorithm
%-------------------------------------------------------------------%
%----Input:
%------training data matrix, test data matrix, parameters
%----Output:
%------diagnostic results:
%------The first row represents the predicted class clabels
%------The second row represents the predicted tumor fraction
[classifier_prediction, sRFDBI_results] = sRFDBI(train_data, test_data);
%-------------------------------------------------------------------%

%% save results
save_path = '../results/';
mkdir(save_path);
save(strcat(save_path,'sRFDBI_results.mat'), 'sRFDBI_results');

%% evaluate deconvolution (simulation dataset only)
if evaluate_deconvolution_flag
    fprintf('Evaluating the deconvolution performance\n');
    [test_MAE,  test_RMSE, test_PCC] = evaluate_deconvolution(sRFDBI_results, test_theta);
    disp(['MAE=' num2str(test_MAE)]);
    disp(['RMSE=' num2str(test_RMSE)]);
    disp(['PCC=' num2str(test_PCC)]);
end

%% evaluate diagnosis
fprintf('Evaluating the diagnostic performance...\n');
average_ACC = evaluate_diagnosis(sRFDBI_results, param);
disp(['average ACC=' num2str(average_ACC)]);

