function hists = compute_spatial_histograms(model, path, imageNames)
% path   (string)           : path of the image files in the file system
% images (array of doubles) : array of image files names without extension 
%                             'png' extension is assumed 
  
  global settings;

  hists = {} ;
  parfor i=1:length(imageNames)
    imagePath = fullfile(path, sprintf('%d.png', imageNames(i)));
    im = imread(imagePath) ;
    hists{i} = get_image_descriptor(model, im);
  end
  hists = cat(2, hists{:});
  
  save(settings.file.hist, 'hists') ;
  disp('Spatial histograms have been computed.');
end
