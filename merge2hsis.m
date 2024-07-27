clear all
clc

data1 = load('results\hsi_5_indoor\I\buildingblock_1ms_renamed_1.mat').data;
data2 = load('results\hsi_5_indoor\I\buildingblock_1ms_renamed_2.mat').data;

data = [];
for i=1:32
    data = cat(3, data, data1(:,:,i));
    data = cat(3, data, data2(:,:,i));
end
