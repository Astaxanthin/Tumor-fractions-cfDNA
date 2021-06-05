function [train_data,test_data] = random_data_split(real_data)

%% randomly split dataset
all_labels = real_data(end,:);
all_class_index = unique(all_labels);
all_methylation_data = real_data(1:(end-1),:);

class_num = zeros(1,size(all_class_index,2));
train_num = zeros(1,size(all_class_index,2));
train_data = [];
test_data = [];
for i = 1:size(all_class_index,2)
    class_num(i) = sum(all_labels==all_class_index(i));
    train_num(i) = round(class_num(i)/2);
    class_data = all_methylation_data(:,all_labels==all_class_index(i));
    rand_index = randperm(class_num(i));
    train_index = rand_index(1:train_num(i));
    test_index = rand_index((train_num(i)+1):end);
    train_data = [train_data,[class_data(:, train_index);repmat(all_class_index(i),1,train_num(i))]];
    test_data = [test_data,[class_data(:, test_index);repmat(all_class_index(i),1,class_num(i) - train_num(i))]];
end
end