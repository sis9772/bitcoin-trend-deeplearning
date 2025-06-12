%% Load processed data
load('data/btc_features.mat');  % Contains X (NxTxF), y (Nx1), mu, sigma

% Check shape
[numSamples, sequenceLength, numFeatures] = size(X);
fprintf("✅ Loaded data shape: %d samples, %d timesteps, %d features\n", numSamples, sequenceLength, numFeatures);

%% Prepare data (transpose for LSTM)
X_seq = cell(numSamples, 1);
for i = 1:numSamples
    X_seq{i} = squeeze(X(i, :, :))';  % F x T
end

Y_seq = categorical(y);

%% Split train/val (80/20)
N = numel(Y_seq);
idx = randperm(N);
nTrain = round(0.8 * N);
XTrain = X_seq(idx(1:nTrain));
YTrain = Y_seq(idx(1:nTrain));
XVal = X_seq(idx(nTrain+1:end));
YVal = Y_seq(idx(nTrain+1:end));

%% Define LSTM network
inputSize = size(X_seq{1}, 1);  % F
numHiddenUnits = 128;
numClasses = numel(categories(Y_seq));

layers = [
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'Verbose', true, ...
    'Plots','training-progress');

%% Train
net = trainNetwork(XTrain, YTrain, layers, options);

%% Confusion matrix
YPred = classify(net, XVal);
figure;
plotconfusion(YVal, YPred);
title('Confusion Matrix - Validation Set');

%% Save prediction
YVal_numeric = double(YVal);
YPred_numeric = double(YPred);
save('prediction_result.mat', 'YVal_numeric', 'YPred_numeric');
fprintf("✅ 예측 결과가 'prediction_result.mat' 파일에 저장되었습니다.\n");