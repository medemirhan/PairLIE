clear all
clc

%% user inputs
inputDir = 'hsi_dataset\outdoor\';
outputDir = 'hsi_dataset\outdoor_rgb\non_normalized\';
normalization = false;
saveMatFile = false;
saveImg = true;
spaces = 3;
totalBands = 204;
startBand = 7;
endBand = 198;

%%
list = ls(strcat(inputDir, '*.mat'));
[len,~] = size(list);

b1 = 400;
b2 = 485;
g1 = 485;
g2 = 590;
r1 = 590;
r2 = 750;

bandsR = [];
bandsG = [];
bandsB = [];

step = (1000-400)/totalBands;

for i=startBand:spaces:endBand
    w1 = 400 + (i-1)*step;
    w2 = w1 + step;

    if w1>r1 && w1<r2
        bandsR = vertcat(bandsR, ((i-(startBand-1))-mod(i-(startBand-1), 3))/3 + 1);
    elseif w1>g1 && w1<g2
        bandsG = vertcat(bandsG, ((i-(startBand-1))-mod(i-(startBand-1), 3))/3 + 1);
    elseif w1>b1 && w1<b2
        bandsB = vertcat(bandsB, ((i-(startBand-1))-mod(i-(startBand-1), 3))/3 + 1);
    end
end

imgR = [];
imgG = [];
imgB = [];

for ff=1:len
    data = load(strcat(inputDir, list(ff,:))).data;
    if normalization
        data = data - min(data(:));
        data = data ./ max(data(:));
    end
    
    for i=1:length(bandsR)
        for j=1:length(bandsG)
            for k=1:length(bandsB)
                imgR = data(:,:,bandsR(i));
                imgG = data(:,:,bandsG(j));
                imgB = data(:,:,bandsB(k));
                img = cat(3, imgR, imgG, imgB);
                parentDir = split(list(ff,:),'.');
                parentDir = parentDir{1,1};
                if normalization
                    parentDir = strcat(parentDir, '_norm');
                end
                savename = strcat(parentDir, '_RGB_', num2str(bandsR(i)), '_', num2str(bandsG(j)), '_', num2str(bandsB(k)));
                savedir = strcat(outputDir, parentDir);
                
                if ~exist(strcat(outputDir, parentDir), 'dir')
                    mkdir(strcat(outputDir, parentDir));
                end
                
                if saveMatFile
                    save(strcat(savedir, '\', savename, '.mat'), 'img');
                end
                
                if saveImg
                    imwrite(img, strcat(savedir, '\', savename, '.png'));
                end
            end
        end
    end
end

    
