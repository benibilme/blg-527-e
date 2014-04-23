function hists = compute_spatial_histograms(path, images)
% path   (string)           : path of the image files in the file system
% images (array of doubles) : array of image files names without extension 
%                             'png' extension is assumed 
  hists = {} ;
  parfor i=1:length(images)
    imageFilePath = fullfile(path, sprintf('%d.png', i));
    im = imread(imageFilePath) ;
    hists{i} = getImageDescriptor(im);
  end
  hists = cat(2, hists{:}) ;
  %save(conf.histFile, 'hists') ;
end
