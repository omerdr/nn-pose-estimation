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

%% Load valid emotions
load('v_emotions')

%% Plots

%% LIBLINEAR
load('liblinear-PrivateTest-predictions')
load('liblinear-PublicTest-predictions')
load('liblinear-Training-predictions')

Results_liblinear_private = PlotResults( testY, liblinear_PrivateTest_predictions, 'liblinear PrivateTest' );
Results_liblinear_public = PlotResults( testY_public, liblinear_PublicTest_predictions, 'liblinear PublicTest ' );
Results_liblinear_training = PlotResults( trainY, liblinear_Training_predictions, 'liblinear Training ' );

%% LIBSVM 15,000 images in the training set
load('libsvm15-PrivateTest-predictions')
load('libsvm15-PublicTest-predictions')
load('libsvm15-Training-predictions')

Results_libsvm15_private = PlotResults( testY, libsvm15_PrivateTest_predictions(:,1), 'libsvm15 PrivateTest' );
Results_libsvm15_public = PlotResults( testY_public, libsvm15_PublicTest_predictions(:,1), 'libsvm15 PublicTest ' );
Results_libsvm15_training = PlotResults( trainY(1:15000), libsvm15_Training_predictions(:,1), 'libsvm15 Training' );

%% LIBSVM 
load('fer_private_test_correct.prediction')
load('fer_public_test.prediction')

Results_libsvm_private = PlotResults( testY, fer_private_test_correct(:,1), 'libSVM PrivateTest' );
Results_libsvm_public = PlotResults( testY_public, fer_public_test(:,1), 'libSVM PublicTest ' );

%% LIBSVM Regularized
load('libsvm-regu-PrivateTest-predictions')
load('libsvm-regu-PublicTest-predictions')
load('libsvm-regu-Training-predictions')
Results_regusvm_private = PlotResults( testY, libsvm_regu_PrivateTest_predictions(:,1), 'libSVM regularized PrivateTest' );
Results_regusvm_public = PlotResults( testY_public, libsvm_regu_PublicTest_predictions(:,1), 'libSVM regularized PublicTest ' );
Results_regusvm_training = PlotResults( trainY, libsvm_regu_Training_predictions(:,1), 'libSVM regularized Training ' );

%% Some more plots
ClassNames = {'Angry', 'Disgust','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'};

fig=figure('Name','Results Comparison','Position',[300, 1000, 1000, 400]);
subplot(211);
bar([Results_libsvm_public.Precision; Results_regusvm_public.Precision; Results_liblinear_public.Precision]');
legend('LIBSVM Public (C=4)', 'Strongly Regularized LIBSVM (C=1)','LIBLINEAR Public','Location','BestOutside');
set(gca,'XTickLabel',ClassNames); title('Precision');

subplot(212);
bar([Results_libsvm_public.Recall; Results_regusvm_public.Recall; Results_liblinear_public.Recall]');
legend('LIBSVM Public (C=4)', 'Strongly Regularized LIBSVM (C=1)', 'LIBLINEAR Public','Location','BestOutside');
set(gca,'XTickLabel',ClassNames); title('Recall');

fig2=figure('Name','Accuracy','Position',[300 600 600 150]);
barh([Results_libsvm_public.Accuracy Results_regusvm_public.Accuracy Results_liblinear_public.Accuracy]);
set(gca,'YTickLabel',{'LIBSVM Public (C=4)', 'Strongly Regularized LIBSVM (C=1)', 'LIBLINEAR Public'}); 
title('Accuracy');

if ~exist(    'figures/precision.eps','file');
    print(fig,'figures/precision','-depsc');
else
    disp('figures/precision.eps exists!');
end;

if ~exist(     'figures/accuracy.eps','file')
    print(fig2,'figures/accuracy','-depsc');
else
    disp('figures/accuracy.eps exists!');
end;

%% Private Tests
fig=figure('Name','Private Test Results Comparison','Position',[300, 1000, 1000, 400]);
subplot(211);
bar([Results_libsvm_private.Precision; Results_regusvm_private.Precision; Results_liblinear_private.Precision]');
legend('LIBSVM (C=4)', 'Strongly Regularized LIBSVM (C=1)','LIBLINEAR','Location','BestOutside');
set(gca,'XTickLabel',ClassNames); title('Private Testset Precision');

subplot(212);
bar([Results_libsvm_private.Recall; Results_regusvm_private.Recall; Results_liblinear_private.Recall]');
legend('LIBSVM (C=4)', 'Strongly Regularized LIBSVM (C=1)', 'LIBLINEAR','Location','BestOutside');
set(gca,'XTickLabel',ClassNames); title('Private Testset Recall');

fig2=figure('Name','Accuracy','Position',[300 600 600 150]);
barh([Results_libsvm_private.Accuracy Results_regusvm_private.Accuracy Results_liblinear_private.Accuracy]);
set(gca,'YTickLabel',{'LIBSVM (C=4)', 'Strongly Regularized LIBSVM (C=1)', 'LIBLINEAR'}); 
title('Private Testset Accuracy');

if ~exist(    'figures/precision-private.eps','file');
    print(fig,'figures/precision-private','-depsc');
else
    disp('figures/precision-private.eps exists!');
end;

if ~exist(     'figures/accuracy-private.eps','file')
    print(fig2,'figures/accuracy-private','-depsc');
else
    disp('figures/accuracy-private.eps exists!');
end;

%% Heat Maps
figure
data = Results_liblinear_private.confusion_matrix;
heatmaptext(data,'fontcolor','w','precision',3);
colormap(redbluecmap)
xlabel(gca,'Prediction'); ylabel('Actual Value');
set(gca,'Visible','on','XTickLabels',ClassNames,'YTickLabels',ClassNames);

