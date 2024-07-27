%%
% Divides HSI overlappingly.
% Ex:
% Band per img = 3
% B1,B2,B3 -> img1
% B3,B4,B5 -> img2
% ...
%%
clear all
clc

%% user inputs
inputDir = 'hsi_dataset';
outputTrainDir = 'train_ll_overlap';
outputTestDir = 'test_ll_overlap';
totalBands = 64;
outputBandNum = 10;
overlap = 4;
reservedForTest = {'007_2_2021-01-20_024_renamed.mat','buildingblock_1ms_renamed.mat'};

%%
if overlap>=outputBandNum
    error('Overlap must be smaller than the output band count');
end

outputTrainDir = strcat(outputTrainDir,'_',num2str(outputBandNum),'_bands');
outputTestDir = strcat(outputTestDir,'_',num2str(outputBandNum),'_bands');
dirs = dir(inputDir);
[len,~] = size(dirs);

dirCount = 1;
for i=3:len
    curDir = strcat(inputDir, '/', dirs(i).name, '/');
    fileList = ls(strcat(curDir, '*.mat'));
    [fileCount, ~] = size(fileList);

    for j=1:fileCount
        curFile = strtrim(fileList(j,:));
        
        filePrefix = split(curFile,'.');
        filePrefix = filePrefix{1,1};
        
        dataOrg = load(strcat(curDir,curFile)).data;
        startBand = 1;
        while totalBands - startBand + 1 >= outputBandNum*2 - overlap
            for k=1:2
                endBand = startBand+outputBandNum-1;
                data = dataOrg(:,:,startBand:endBand);
                generatedImgName = filePrefix;
                for m=startBand:endBand
                    generatedImgName = strcat(generatedImgName,'_',num2str(m));
                end
                generatedImgName = strcat(generatedImgName, '.mat');
                
                if any(strcmp(reservedForTest, curFile))
                    for m=1:length(reservedForTest)
                        if strcmp(reservedForTest(m), curFile)
                            testDirNum = m;
                            break;
                        end
                    end
                    savePath = fullfile(outputTestDir,num2str(testDirNum));
                else
                    savePath = fullfile(outputTrainDir, num2str(dirCount));
                end

                if ~exist(savePath, 'dir')
                    mkdir(savePath);
                end
                save(fullfile(savePath,generatedImgName), 'data');
                startBand = endBand - overlap + 1;
            end
            if ~any(strcmp(reservedForTest, curFile))
                dirCount = dirCount + 1;
            end
        end
    end
end


