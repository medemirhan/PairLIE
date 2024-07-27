%%
% Merges overlapping HSIs.
% Ex:
% Band per img = 3
% B1,B2,B3 -> img1
% B3,B4,B5 -> img2
% ...
% Merge: B1,B2,B3,...Bn
%%
clear all
clc

%% user inputs
inputDir = 'results\test_ll_overlap_3_bands\1\I';
outputDir = 'train_ll_overlap';
totalBands = 64;
outputBandNum = 3;
overlap = 1;

%%
if overlap>=outputBandNum
    error('Overlap must be smaller than the output band count');
end

fileList = ls(strcat(inputDir, '/*.mat'));
[len,~] = size(fileList);

data = [];
for i=1:len
    curFile = strtrim(fileList(i,:));
    filePrefix = split(curFile,'.');
    filePrefix = filePrefix{1,1};

    dataOrg = load(fullfile(inputDir,curFile)).data;

    if i == 1
        data = dataOrg;
    else
        for j=1:overlap
            temp = (dataOrg(:,:,j) + data(:,:,outputBandNum-overlap+j))/2;
            data(:,:,outputBandNum-overlap+j) = temp;
        end
        data = cat(3, data, dataOrg(:,:,overlap+1:outputBandNum));
    end

end

% over=2
% out=5
% 1 2 3 4 5
% 4 5 6 7 8




