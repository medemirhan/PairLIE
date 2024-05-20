clear all
clc

%% user inputs
inputDir = 'train_ll_divided_in_the_mid';
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
    data = dataOrg(:,:,1:(totalBands/2));
    save(strcat(curDir, filePrefix, '_1.mat'), 'data');
    data = dataOrg(:,:,(totalBands/2 + 1):totalBands);
    save(strcat(curDir, filePrefix, '_2.mat'), 'data');
end
