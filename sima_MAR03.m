%% load file
% labeling
load('joints.mat');
load('features.mat');

%% indexes
num_images = size(features,1);
train_idx = 1:floor(num_images/2);
test_idx = train_idx(end)+1:num_images;


%% dimension reduction
X = features;
[U,S,V] = svd(X);
k=1000;
Sk=S(:,1:k);
Uk=U;%(1:k,1:k);
Vk=V(1:k,1:k);

Xk=Uk*Sk*Vk';
Xtrain=double(X(train_idx));
Xtest=double(X(test_idx));

%% regression
Y=squeeze(joints(1,14,train_idx)); %the head
beta = mvregress(Xk',Y);

%% PCA
% first, 0-mean data
% Xtrain = bsxfun(@minus, Xtrain, mean(Xtrain,1));           
% Xtest  = bsxfun(@minus, Xtest, mean(Xtrain,1));           
% 
% % Compute PCA
% covariancex = (Xtrain'*Xtrain)./(size(Xtrain,1)-1);                 
% [V D] = eigs(covariancex, 900);   % reduce to 10 dimension
% 
% pcatrain = Xtrain*V;
% % here you should train your classifier on pcatrain and ytrain (correct labels)
% 
% pcatest = Xtest*V;
% % here you can test your classifier on pcatest using ytest (compare with correct labels)

