clc;
clear;
close all;
%%

% USE THIS SCRIPT TO (+Camera_calbration.mat) TO UNDISTORT YOUR WAVEGUIDE TEST RESULTS

load Camera_calibration.mat %load camera calibration parameters
% load up file path
imageDir = 'E:\Mac\20sdwell_0pt1mmpsretraction\5N\Distorted Split Frames';

% Save results to this directory
Save_dir = 'E:\Mac\20sdwell_0pt1mmpsretraction\5N\Undistorted Split Frames';

% List of all .png images in the directory
images = dir(fullfile(imageDir, '*.png'));

num_images = numel(images);
% magnification = 40;

% Select the images you want to read into the script
for i = 1:num_images
    % Read the Image of Objects to Be Measured
    image_name = images(i).name; % Use the actual image name from the directory
    fullImageName = fullfile(imageDir, image_name);
    imOrig = imread(fullImageName);
    % imOrig=imread(image_name);

%     imOrig {i} = imread('frame_000013.png');
%     figure %('Visible','off'); 
%     imshow(imOrig{i}, 'InitialMagnification', magnification);
%     title('Input Image');
    
    % Undistort the Image
    [im, newOrigin] = undistortImage(imOrig, cameraParams, 'OutputView', 'valid');
    %f = figure('Visible','off');
    %imshow(im{i}, 'InitialMagnification', magnification);
    % title('Undistorted Image');
    
    %save images
    fullFileName= fullfile(Save_dir, image_name);
    imwrite(im,fullFileName,'png')
    %imshow(img)
    %close(f)
end