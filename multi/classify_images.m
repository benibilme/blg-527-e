function classifications = classify_images(model, path, imageNames)
% path : path of the image file
% imageNames : Names of the images without extension 
  classifications = {};
  parfor i=1:length(imageNames)
    fileName = sprintf('%d.png',imageNames(i));
    im = imread(fullfile(path, fileName));
    [className, classId] = classify_image(model, im);
    classifications{i} = className;
  end
  save('classes.mat', 'classifications');
  disp('Images have been classified');
end