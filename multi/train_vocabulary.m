function vocab = train_vocabulary(path, images, phowOpts, numWords)  
% path   (string)           : path of the image files in the file system
% images (array of doubles) : array of image files names without extension 
%                             'png' extension is assumed 
% options(cell array) : whose members are ordered list of options 
%                       value for vl_phow 

  global settings;

  descrs = {} ;
  parfor i=1:length(images)
    imagePath = fullfile(path, sprintf('%d.png',images(i)));
    im = imread(imagePath) ;
    im = get_standart_image(im) ;
    [~, descrs{i}] = vl_phow(im, phowOpts{:});
  end
  descrs = cat(2, descrs{:});
  %descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;
  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, numWords, ...
                    'algorithm', 'elkan', ...
                    'MaxNumIterations', 50);
  
  save(settings.file.vocab, 'vocab');
  disp('Vocabulary has been trained');
end