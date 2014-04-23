function classifications = classify_images(path, imageNames)
% path : path of the image file
% imageNames : Names of the images without extension 
  classifications = {};
  parfor i=1:length(imageNames)
    fileName = sprintf('%d.png',imageNames(i));
    im = imread(fullfile(path, fileName));
    [className, classId] = classify_image(im);
    classifications{i} = className;
  end
  save('classes.mat', 'classifications');
end