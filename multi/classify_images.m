function classifications = classify_images(model, data, path)
% path : path of the image file
% imageNames : Names of the images without extension 
  classifications = {};
  parfor i=1:length(data.imageFileNames)
    fileName = sprintf('%d.png',data.imageFileNames(i));
    im = imread(fullfile(path, fileName));
    [className, classId] = classify_image(model, data.imageClassSet, im);
    classifications{i} = className;
  end
  save('classes.mat', 'classifications');
  disp('Images have been classified');
end