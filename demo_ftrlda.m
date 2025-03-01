testRatio=0.5;   trainRatio=1-testRatio;
K=1;

for g = 1:10
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                        yale cmu 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

data_name = 'yale';  % yale
rootFolder = fullfile('data', data_name); 
%% size
Width = 80;
Height = 60;

TrainData=[];
TrainLabels=[];
TestData=[];
TestLabels=[];
subfolders = dir(rootFolder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(3:end); 
for classIndex = 1:numel(subfolders)
    imageData = [];
    labels = [];
    classFolder = fullfile(rootFolder, subfolders(classIndex).name);
       imageFiles = [dir(fullfile(classFolder, '*.jpg')); ...
              dir(fullfile(classFolder, '*.png')); ...
              dir(fullfile(classFolder, '*.pgm'))];
    for imageIndex = 1:numel(imageFiles)
        imagePath = fullfile(classFolder, imageFiles(imageIndex).name);
        image = imread(imagePath);
        if size(image, 3) == 3
            image = rgb2gray(image);
        end
        image = imresize(image, [Height, Width]);
        image=double(image);
        imageData = [imageData; image(:)'];
        labels = [labels; classIndex];
    end
    permutedIdx = randperm(length(labels));
    imageData = imageData(permutedIdx, :);
    labels = labels(permutedIdx);
    numData = size(imageData, 1);
    numTrainData = round(trainRatio * numData);
    counter(classIndex)=numTrainData;
    numTestData = numData - numTrainData;
    trainData = imageData(1:numTrainData, :);
    testData = imageData(numTrainData + 1:numData, :);
    trainLabels = labels(1:numTrainData);
    testLabels = labels(numTrainData + 1:numData);
    TrainData=[TrainData;trainData];
    TrainLabels=[TrainLabels;trainLabels];
    TestData=[TestData;testData];
    TestLabels=[TestLabels;testLabels];
end
%%%%% hw  hb %%%%%%%
X=TrainData'; NumTrainData=size(X,2); hb=[]; hw=[];
mean=sum(X,2)/NumTrainData; 
for i=1:classIndex
    if i~=1 
        sta=sum(counter(1:i-1))+1; 
    else
        sta=1;
    end
    Ai=X(:,sta:sta-1+counter(i));
    meani=sum(Ai,2)/counter(i);
    ei=double(ones(1,counter(i))); 
    hb=[hb,sqrt(counter(i))*(meani-mean)];
    hw=[hw,Ai-meani*ei];
end
dim=size(X,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                   wine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %%    wine
% data = load('data/wine/wine.txt');
% labels=data(:,1);
% data =data(:,2:end);
% 
% 
% ratio=testRatio;
% [~,label] = hist(labels, unique(labels));
% classIndex=numel(label);
% TrainData=[];TestData=[];TrainLabels=[];TestLabels=[];hb=[];hw=[];
% av=mean(data,1);
% for i=1:classIndex
% cls_idx = (labels == label(i));
% datai=data(cls_idx,:);
% num_samples = size(datai, 1);
% random_indices = randperm(num_samples);
% datai = datai(random_indices, :);
% traindata=datai(1:round(ratio*num_samples),:);
% testdata=datai(round(ratio*num_samples)+1:end,:);
% numi=size(traindata,1);
% trainlabels=i*ones(numi,1);
% testlables=i*ones(size(testdata,1),1);
% TrainData=[TrainData;traindata];
% TrainLabels=[TrainLabels;trainlabels];
% TestData=[TestData;testdata];
% TestLabels=[TestLabels;testlables];
% meani=mean(traindata,1);
% hb=[hb,sqrt(numi)*(meani-av)'];
% ei=ones(1,numi);
% hw=[hw,traindata'-meani'*ei];
% end
% rb=rank(hb);
% rw=rank(hw);
% rh=rank([hb';hw']);
% NumTrainData=size(TrainData,1);
% num=NumTrainData;
% [counter,~] = hist(TrainLabels, unique(TrainLabels));


%% TRLDA
%% Subspace
dem=45;  %yale=45  cmu=41  wine=6    

tic; 
sb=hb*hb';sw=hw*hw'; 
O=randn(dim,dem); [W0,~]=qr(O,0);
W=zeros(size(W0)); t=1; fun(1)=0; I=eye(dim); aer=0; 
while norm(W0-W,'fro')>1e-4
W=W0; st=sqrt(trace(W0'*sb*W0))/trace(W0'*sw*W0); 
St=-st^2*sw;
Bt=st*sb*W0/sqrt(trace(W0'*sb*W0));
M=2*(St*W0+Bt); 
if t==1
Q=myqr(M,1e-3,round((dem)/2)); 
Im=eye(dem); BQ=size(Q,2);
Z=zeros(BQ,dem-BQ); 
IBQ=eye(BQ);
I1=[IBQ,Z];
end
[U,~,V]=svd(Q'*M); 
W0=Q*U*I1*V'; 
t=t+1;
fun(t)=st^2*trace(W0'*sw*W0)-2*st*sqrt(trace(W0'*sb*W0));

if abs(fun(t)-fun(t-1))<1e-4
    break
end
end
touying=W0; 
trlda_time(g)=toc;
%%
X0=TrainData*touying;
Y0=TrainLabels;
X1=TestData*touying;
Y1=TestLabels;
knn_classifier = fitcknn(X0, Y0, 'NumNeighbors', K);
predicted_labels = predict(knn_classifier, X1);
accuracy = sum(predicted_labels == Y1) / numel(Y1);
fprintf('KNN分类准确率: %.2f%%\n', accuracy * 100);
lda_accuracy(g)=accuracy;
end
accuracy=sum(lda_accuracy)/g*100;
dev=std(lda_accuracy)*100;
time=sum(trlda_time)/g;
fprintf('平均准确率：%.2f%', accuracy);fprintf('±%.2f%', dev);
fprintf('\n平均时间：%.4f%', time);