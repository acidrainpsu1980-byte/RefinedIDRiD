classNames = ["Background" "Retina" "FV" "Vessel" "OD" "VH" "EX" "IRMA" "HE" "NV" "CWS" "MA"];
labelIDs   = [0 8 16 24 32 4 63 96 127 166 191 255];
% Create datastore
imds = imageDatastore('Data\Train\Images');
pxds = pixelLabelDatastore('Data\Train\Labels', classNames, labelIDs);

%Create learning model.
imageSize = [1024 1024 3];
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50");

%Select Training Options
imdsVal = imageDatastore('Data\Test\Images');
pxdsVal = pixelLabelDatastore('Data\Test\Labels', classNames, labelIDs);
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);
maxEpochs = 100;
options = trainingOptions('sgdm', ...
    'Momentum', 0.95, ...
    'InitialLearnRate', 0.35e-2, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',5,...
    'LearnRateDropFactor',0.9,...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', maxEpochs, ...  
    'ExecutionEnvironment','gpu',...
      'ValidationData',pximdsVal,...
    'ValidationFrequency',100,...
    'MiniBatchSize', 3, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'VerboseFrequency', 100);

%Data Augmentation
augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation', [-10 10], 'RandRotation', [-180 180]);
dsTrain = pixelLabelImageSource(imds,pxds,...
   'DataAugmentation',augmenter);
% Train model
[net, info] = trainNetwork(dsTrain,lgraph,options);
modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
save(['RefinedIDRiD_Rn50-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net','options', 'info');

%Test Network on One Image
cmap = [0 0 0;                  %Background
    0.7529	0.3764	0.1254  %Retina
    1 0  0                              %FV
    0 1 0                               %Vessel
    1 1 1                               %OD
    0.7 0.4 1                         %VH
1	1	0.0667                        %EX
0.7	0.7	0.7                       %IRMA
0	1	1                                  %HE
0.7  0.2 0                              %NV
0  0  1                                   %CWS
1	0	1];                                   %MA
N = numel(classNames);
ticks = 1/(N*2): 1/N : 1;
idx = 1;
I = readimage(imdsVal,idx);
C = semanticseg(I, net);

B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
GT = readimage(pxdsVal, idx);
D = labeloverlay(I, GT, 'Colormap', cmap, 'Transparency',0.4);
figure, montage({I, B, D}, "Size", [1 3])
pixelLabelColorbar(cmap, classNames);
%Support functions
function pixelLabelColorbar(cmap, classNames)
    % Add a colorbar to the current axis. The colorbar is formatted
    % to display the class names with the color.
    
    colormap(gca,cmap)
    
    % Add colorbar to current figure.
    c = colorbar('peer', gca);
    
    % Use class names for tick marks.
    c.TickLabels = classNames;
    numClasses = size(cmap,1);
    
    % Center tick labels.
    c.Ticks = 1/(numClasses*2):1/numClasses:1;
    
    % Remove tick mark.
    c.TickLength = 0;
end