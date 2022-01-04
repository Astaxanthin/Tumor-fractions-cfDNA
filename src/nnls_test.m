function [test_V, pre_labels] = nnls_test(test_data, train_U, param)
seed = 666;
rand('seed',seed);

class_num=param.class_num;
healthy_pattern_num = param.healthy_pattern_num;
cancer_pattern_num = param.cancer_pattern_num;
r=healthy_pattern_num + (class_num-1)*cancer_pattern_num;
n2=size(test_data,2);

test_V = zeros(r,n2);

if ~param.weighted
    W_value = ones(1,size(train_U,1));
else
    W_value= param.marker_weight';
    W_value(W_value<=0) = 1e-5;
end

p = param.p;
pre_labels = zeros(1,n2);
if size(test_data,1)<1000
    parfor i = 1:size(test_data,2)
        i
        err = [];
        aval_x = zeros(r ,class_num);
        %     for k = 1:class_num
        k = param.pseudo_label(i);
        if k ==0
            mask =zeros(1,r);
        elseif k ==1
            mask = [zeros(1,healthy_pattern_num),ones(1,cancer_pattern_num*(class_num-1))];
        else
            mask = [zeros(1,healthy_pattern_num),ones(1,cancer_pattern_num*(class_num-1))];
            mask((healthy_pattern_num+cancer_pattern_num*(k-2)+1):(healthy_pattern_num+cancer_pattern_num*(k-1))) = 0;
        end
        
        fun = @(x)norm(W_value'.*(test_data(:,i)-train_U*x),'fro');
        %sum((W_value'.*(test_data(:,i)-train_U*x)).^2).^(p/2);
        x0 = rand([r,1]);
        A = [];
        b = [];
        Aeq = [ones(1,r);mask];
        beq = [1,0];
        %         Aeq = ones(1,r);
        %         beq = 1;
        lb = zeros(1,r);
        ub = ones(1,r);
        %         nonlcon = @nonlinear;
        options = optimoptions('fmincon','Display','none');
        test_V(:,i) = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,[],options);
        %     aval_x(:,k) = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        
        %     err(k) = norm(test_data(:,i)-train_U*aval_x(:,k),'fro');
    end
    % [~,min_index] = sort(err);
    % test_V(:,i) = aval_x(:,min_index(1));
    
else
    for i = 1:size(test_data,2)
        i
        err = [];
        aval_x = zeros(r ,class_num);
        %     for k = 1:class_num
        k = param.pseudo_label(i);
        if k ==0
            mask =zeros(1,r);
        elseif k ==1
            mask = [zeros(1,healthy_pattern_num),ones(1,cancer_pattern_num*(class_num-1))];
        else
            mask = [zeros(1,healthy_pattern_num),ones(1,cancer_pattern_num*(class_num-1))];
            mask((healthy_pattern_num+cancer_pattern_num*(k-2)+1):(healthy_pattern_num+cancer_pattern_num*(k-1))) = 0;
        end
        
        fun = @(x)norm(W_value'.*(test_data(:,i)-train_U*x),'fro');
        %sum((W_value'.*(test_data(:,i)-train_U*x)).^2).^(p/2);
        x0 = rand([r,1]);
        A = [];
        b = [];
        Aeq = [ones(1,r);mask];
        beq = [1,0];
        %         Aeq = ones(1,r);
        %         beq = 1;
        lb = zeros(1,r);
        ub = ones(1,r);
        %         nonlcon = @nonlinear;
        options = optimoptions('fmincon','Display','none');
        test_V(:,i) = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,[],options);
        %     aval_x(:,k) = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        
        %     err(k) = norm(test_data(:,i)-train_U*aval_x(:,k),'fro');
    end
    % [~,min_index] = sort(err);
    % test_V(:,i) = aval_x(:,min_index(1));
    
end

end

% for i = 1:size(test_data,2)
%     i
%     err = inf*ones(1,class_num);
%     test_err= inf*ones(1,class_num);
%     aval_x = zeros(r ,class_num);
%     for k = 1:class_num
%         %k = pseudo_label(i);
%         %             if k_label ==0
%         %                 mask =zeros(1,r);
%         if k ==1
%             mask = [zeros(1,healthy_pattern_num),ones(1,cancer_pattern_num*(class_num-1))];
%         else
%             mask = [zeros(1,healthy_pattern_num),ones(1,cancer_pattern_num*(class_num-1))];
%             mask((healthy_pattern_num+cancer_pattern_num*(k-2)+1):(healthy_pattern_num+cancer_pattern_num*(k-1))) = 0;
%         end
%
%         fun = @(x)norm(W_value'.*(test_data(:,i)-train_U*x),'fro');
%         %sum((W_value'.*(test_data(:,i)-train_U*x)).^2).^(p/2);
%         x0 = rand([r,1]);
%         A = [];
%         b = [];
%         Aeq = [ones(1,r);mask];
%         beq = [1,0];
%         %         Aeq = ones(1,r);
%         %         beq = 1;
%         lb = zeros(1,r);
%         ub = ones(1,r);
%         %         nonlcon = @nonlinear;
%         options = optimoptions('fmincon','Display','none');
%         %             test_V(:,i) = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,[],options);
%         aval_x(:,k) = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,[],options);
%
%         err(k) = norm(test_data(:,i)-train_U*aval_x(:,k),'fro');
%         test_err(k) = std((test_data(:,i)-train_U*aval_x(:,k)).^2);
%     end
%     [~,label] = min(err);
%     [~,test_label] = min(test_err);
%     test_V(:,i) = aval_x(:,test_label);
%     pre_labels(i) = test_label;
%     abs(err(1)-mean(err(2:end)))
% end