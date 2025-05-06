%% RESET 
clc;
clear all;

%% LOAD DATA
imageFolder = 'C:\Users\matsk\OneDrive\Pulpit\Notes\VI_SEMESTER\AO\Labs\Lab2\data';

imageFiles = dir(fullfile(imageFolder, '*.bmp')); 
numImages = length(imageFiles);
images = cell(1, numImages);

for k = 1:numImages
    filename = fullfile(imageFolder, imageFiles(k).name);
    images{k} = imread(filename);
end

%% CONVERT TO HSV

fhsvImg = cell(numImages, 1);
hue = cell(numImages, 1);
sat = cell(numImages, 1);
val = cell(numImages, 1);

for k = 1:numImages
    fhsvImg{k} = rgb2hsv(images{k});
    hue{k} = fhsvImg{k}(:,:,1);
    sat{k} = fhsvImg{k}(:,:,2);
    val{k} = fhsvImg{k}(:,:,3);
end

%% DETECT RED REGION & MEASURE AREA

redMask = cell(numImages, 1);       % combined masks
cleanMask = cell(numImages, 1);     % cleaned binary masks
areaPixels = zeros(numImages, 1);   % area values

for k = 1:numImages
    % threshold
    redMask1 = (hue{k} >= 0.0 & hue{k} <= 0.05) & sat{k} > 0.35 & val{k} > 0.15;
    redMask2 = (hue{k} >= 0.95 & hue{k} <= 1.0) & sat{k} > 0.35 & val{k} > 0.15;
    redMask{k} = redMask1 | redMask2;

    % fill the holes
    maskFilled = imfill(redMask{k}, 'holes');
    maskCleaned = bwareaopen(maskFilled, 100); 
    cleanMask{k} = maskCleaned;

    % measure area in pixels
    areaPixels(k) = sum(maskCleaned(:));
end

%% SHOWING RESULTS

for k = 1:numImages
    figure('Name', ['Image ', num2str(k)], 'NumberTitle', 'off');

    % original img
    subplot(1,2,1);
    imshow(images{k});
    title('Original');

    % overlayed img
    subplot(1,2,2);
    imshow(images{k});
    hold on;

    % blue overlay
    blueOverlay = cat(3, zeros(size(cleanMask{k})), zeros(size(cleanMask{k})), ones(size(cleanMask{k})));
    h = imshow(blueOverlay);
    set(h, 'AlphaData', 0.4 * cleanMask{k});

    title(['Overlayed (Blue Mask) — Area: ', num2str(areaPixels(k)), ' px']);
    hold off;
end

%% SAVE RESULTS

imageNames = {imageFiles.name}';
resultTable = table(imageNames, areaPixels, 'VariableNames', {'Filename', 'RedAreaPixels'});
disp(resultTable);


%% BAR CHART

figure;
bar(resultTable.RedAreaPixels, 'FaceColor', [0.2 0.6 1]);
xticks(1:height(resultTable));
xticklabels(resultTable.Filename);
xtickangle(45);
ylabel('Red Area [pixels]');
xlabel('Image Filename');
title('Red-Filled Circle Area per Image');
grid on;

%% LINE PLOT

figure;
plot(resultTable.RedAreaPixels, '-o', 'LineWidth', 2, 'Color', [0.1 0.4 0.8]);
xticks(1:height(resultTable));
xticklabels(resultTable.Filename);
xtickangle(45);
ylabel('Red Area [pixels]');
xlabel('Image Filename');
title('Trend of Red Area across Images');
grid on;

%% PIE CHART

figure;
pie(resultTable.RedAreaPixels, resultTable.Filename);
title('Proportional Red Area per Image');