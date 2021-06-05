function [classifier_prob, classifier_prediction, sRFDBI_prob, sRFDBI_prediction] = BI(train_data, test_data, method, coarse_train_H, coarse_test_H, param)

gt_label = [];
for i = 1:param.class_num
    gt_label =  [gt_label, i*ones(1,param.train_sample_num(i))];
end

if strcmp(method, 'RF')  % Random forest
    model = TreeBagger(100, train_data', gt_label', 'Method','classification');
    [pre_results,classifier_prob] = predict(model,test_data');
    classifier_prediction = zeros(size(pre_results,1),1);
    for i =1:size(pre_results,1)
        classifier_prediction(i) = str2num(pre_results{i,1});
    end
    
elseif strcmp(method, 'SVM')  % Support vector machine
    t = templateSVM('Standardize',1,'KernelFunction','linear');
    options = statset('UseParallel',true);
    SVMModel = fitcecoc(train_data',gt_label,'Options',options,'Learners',t, 'FitPosterior',true,'Holdout',0.15);
    Mdl = SVMModel.Trained{1};
    [classifier_prediction,neg_loss, pre_score, classifier_prob] = predict(Mdl,test_data','Options',options);
    
elseif strcmp(method, 'NN')  % Neural networks.
    
    gt_onehot = zeros(param.class_num,size(train_data,2));
    for i = 1:size(train_data,2)
        gt_onehot(gt_label(i),i) = 1;
    end
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
    
    % Create a Pattern Recognition Network
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    % Train the Network
    [net,tr] = train(net,train_data,gt_onehot);
    
    classifier_prob = sim(net,test_data);
    [~,classifier_prediction] = max(classifier_prob,[],1);
    classifier_prob = classifier_prob';
    classifier_prediction = classifier_prediction';
    
elseif strcmp(method, 'Equal')  %% Equal prior
    classifier_prob = repmat(1/param.class_num,size(test_data,2),param.class_num);%ones(size(test_data,2),param.class_num)/param.class_num; 
    classifier_prediction = randi(param.class_num,  size(test_data,2), 1);    
end
classifier_prediction = classifier_prediction';

%% Bayesian inference
sample_num = [0,param.train_sample_num];
theta_dist = {};
for i = 1:param.class_num
    train_burden = coarse_train_H((param.healthy_pattern_num+1):end, (sum(sample_num(1:i)) + 1):sum(sample_num(1:(i+1))));
    for j = 1:size(train_burden,1)
        train_theta_one = train_burden(j,:);
        beta_ab = betafit(train_theta_one);
        theta_dist{i}(j,:) = beta_ab;
    end
end

sRFDBI_prob = classifier_prob';
for i = 1:size(coarse_test_H,2)
    for j = 1:param.class_num
        for k = 1:size(theta_dist{j},1)
            beta_param = theta_dist{j}(k,:);
            likelyhood = betapdf(coarse_test_H(param.healthy_pattern_num+k,i),beta_param(1),beta_param(2));
            sRFDBI_prob(j,i) = sRFDBI_prob(j,i)*likelyhood;
        end
    end
    sRFDBI_prob(:,i) = sRFDBI_prob(:,i)/sum(sRFDBI_prob(:,i));
end

[bayes_p, sRFDBI_prediction] = max(sRFDBI_prob,[],1);

end