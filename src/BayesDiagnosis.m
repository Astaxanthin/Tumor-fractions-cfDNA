function [posterior, pre_labels, bayes_prob, bayes_labels, save_Model] = BayesDiagnosis(train_data, gt_label, test_data, method, train_V_without_label, test_V_without_label, param, j)


if strcmp(method, 'RF')
    
    % RF training
%     model = TreeBagger(100, train_data', gt_label', 'Method','classification');
    
    % load example models
    load(strcat('./data/real_data/Chen_data/late_stage_training/result/enh/RF/example_models/RFmodel_',num2str(j)));
    model = save_model;
    
    [pre_results,posterior] = predict(model,test_data');
    pre_labels = zeros(size(pre_results,1),1);
    for i =1:size(pre_results,1)
        pre_labels(i) = str2num(pre_results{i,1});
    end
    save_Model = model;
    
elseif strcmp(method, 'SVM')
    t = templateSVM('Standardize',1,'KernelFunction','linear');
    options = statset('UseParallel',true);
    
    % SVM training
%     SVMModel = fitcecoc(train_data',gt_label,'Options',options,'Learners',t, 'FitPosterior',true,'Holdout',0.15);
%     Mdl = SVMModel.Trained{1};

    % load example models
    load(strcat('./data/real_data/Chen_data/late_stage_training/result/enh/SVM/example_models/SVMmodel_',num2str(j)));
    Mdl = save_model;

    [pre_labels,neg_loss, pre_score, posterior] = predict(Mdl,test_data','Options',options);
    save_Model = Mdl;
    
elseif strcmp(method, 'MLP')
    
    % MLP training
%     gt_onehot = zeros(param.class_num,size(train_data,2));
%     for i = 1:size(train_data,2)
%         gt_onehot(gt_label(i),i) = 1;
%     end
%     trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
%     
%     % Create a Pattern Recognition Network
%     hiddenLayerSize = 10;
%     net = patternnet(hiddenLayerSize, trainFcn);
%     
%     % Setup Division of Data for Training, Validation, Testing
%     net.divideParam.trainRatio = 70/100;
%     net.divideParam.valRatio = 15/100;
%     net.divideParam.testRatio = 15/100;
%     
%     % Train the Network
%     [net,tr] = train(net,train_data,gt_onehot);
    
    % load example models
    load(strcat('./data/real_data/Chen_data/late_stage_training/result/enh/MLP/example_models/MLPmodel_',num2str(j)));
    net = save_model;
    
    posterior = sim(net,test_data);
    [~,pre_labels] = max(posterior,[],1);
    posterior = posterior';
    pre_labels = pre_labels';
    
    save_Model = net;
    
elseif strcmp(method, 'Naive')
    
    posterior = repmat(1/param.class_num,size(test_data,2),param.class_num);%ones(size(test_data,2),param.class_num)/param.class_num; 
    pre_labels = randi(param.class_num,  size(test_data,2), 1);
    
end

%% Bayes 
class_type = unique(gt_label);
theta_dist = {};
for i = 1:size(class_type,2)
    train_burden = train_V_without_label((param.healthy_pattern_num+1):end, gt_label == class_type(i));
%     tumor_fraction = sum(train_burden,1);
%     train_burden = [train_burden; tumor_fraction];
    for j = 1:size(train_burden,1)
        train_theta_one = train_burden(j,:);
        beta_ab = betafit(train_theta_one);
        theta_dist{i}(j,:) = beta_ab;
    end
end

bayes_prob = posterior';
test_burden = test_V_without_label((param.healthy_pattern_num+1):end,:);
% test_burden = [test_burden; sum(test_burden,1)];
for i = 1:size(test_V_without_label,2)
    for j = 1:size(class_type,2)
        for k = 1:size(theta_dist{j},1)
            beta_param = theta_dist{j}(k,:);
            likelyhood = betapdf(test_burden(k,i),beta_param(1),beta_param(2));
            bayes_prob(j,i) = bayes_prob(j,i)*likelyhood;
        end
    end
    bayes_prob(:,i) = bayes_prob(:,i)/sum(bayes_prob(:,i));
end

[bayes_p, bayes_labels] = max(bayes_prob,[],1);

end