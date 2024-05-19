clear all
clc

%% user inputs
inputDir = 'train_ll';
totalBands = 64;

%%
dirs = dir(inputDir);
[len,~] = size(dirs);

for i=3:len
    curDir = strcat(inputDir, '/', dirs(i).name, '/');
    file = ls(strcat(curDir, '*.mat'));
    filePrefix = split(file,'.');
    filePrefix = filePrefix{1,1};
    
    dataOrg = load(strcat(curDir,file)).data;
    data = dataOrg(:,:,1:2:(totalBands-1));
    save(strcat(curDir, filePrefix, '_1.mat'), 'data');
    data = dataOrg(:,:,2:2:totalBands);
    save(strcat(curDir, filePrefix, '_2.mat'), 'data');
end
