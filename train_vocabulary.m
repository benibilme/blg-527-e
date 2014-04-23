function vocab = train_vocabulary(path, images, options)  
% path   (string)           : path of the image files in the file system
% images (array of doubles) : array of image files names without extension 
%                             'png' extension is assumed 
% options(cell array) : whose members are ordered list of options 
%                       value for vl_phow 
  descrs = {} ;
  parfor i=1:length(images)
    imageFilePath = fullfile(path, sprintf('%d.png',i));
    im = imread(imageFilePath) ;
    im = standarizeImage(im) ;
    [~, descrs{i}] = vl_phow(im, options);
  end
  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;
  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, model.numWords, ...
                    'algorithm', 'elkan', ...
                    'MaxNumIterations', 50);
  %save(conf.vocabFile, 'vocab');
end