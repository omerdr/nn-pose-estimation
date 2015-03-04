% cd code
% cd github/nn-pose-estimation/
load features.mat
load joints.mat
X = features(1:1900,:); Xtest = features(1901:end,:);
Y = reshape(joints(:,:,1:1900),3*14,1900);
Y = Y';

Ytest = reshape(joints(:,:,1901:end),3*14,100);
Ytest = Ytest';

X = double(X); Xtest = double(Xtest);
%models = cell(1,size(Ytest,2)); % A SVM model for every joint
estimates = zeros(size(Ytest));

for i = 1:size(Ytest,2)
    %models{i} = svmtrain(Y(1:1900,i),X,'-s 3')
    models = svmtrain(Y(1:1900,i),X,'-s 3')
    [prediction, mse, prob_est] = svmpredict(Ytest(1:100,i), Xtest, models );%{i});
    estimates(:,i) = prediction;
    display(num2str(i));
end;

est_joints = reshape(estimates',[3 14 100]);
for i=1:100
    f = figure;
    im = imread(['~/Downloads/lsp_dataset_original/im' num2str(1900+i) '.jpg']);
    imshow(im); hold;
    plot(joints(1,:,1900+i),joints(2,:,1900+i),'*');
    plot(est_joints(1,:,i),est_joints(2,:,i),'*r');
    pause;
    close(f);
end;

% X1 = [ ones(size(X,1),1) X ];
% Xcell = { X1 };
% % mvregress(Xcell, Y); % Fails: 2000 samples aren't enought to fit 4096 parameters
% imagesc(X)
% impixelregion(imagesc(X))
% 
% svmstruct = svmtrain(X, Y(:,3),'ShowPlot',true);