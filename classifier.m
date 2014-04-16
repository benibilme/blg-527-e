% YAPILACAKLAR
% TEN FOLD CROSSVALIDATION GORE KOD DUZENELENECEK
% CLASSIFIER PERFORMANCE PARAMETRELERI CIKARILACAK ROUTINE TAMAMLANACAK
% 

function classifier()

  global conf trainingData model;

  conf.clobber         = true;
  conf.trainDataPath   = 'train';
  conf.testDataPath    = 'test';
  conf.vocabFile       = 'vocab.mat';
  conf.histFile        = 'hists.mat';
  conf.modelFile       = 'model.mat';
  conf.classFile       = 'class.mat';
  conf.dataFile        = 'trainingData.mat';
  conf.fmapFile        = 'fmap.mat';
  conf.cpFile          = 'cp.mat';

  randn('state', 1) ;
  rand('state', 1) ;
  vl_twister('state', 1) ;

  % --------------------------------------------------------------------
  %  Setup trainingData
  % --------------------------------------------------------------------

  trainingData  =  importData('trainLabels.csv');
  save(conf.dataFile, 'trainingData');
  disp('trainingData setup is completed');

  % --------------------------------------------------------------------
  %  Setup up model parameters
  % --------------------------------------------------------------------

  model.numWords = 600;
  model.phowOpts = {'Step', 3};
  model.numSpatialX = [2 4];
  model.numSpatialY = [2 4];
  model.quantizer = 'kdtree';
  model.vocab = [] ;
  model.w = [] ;
  model.b = [] ;
  model.svm.C = 10 ;
  model.svm.solver = 'sdca' ;
  %model.svm.solver = 'sgd' ;
  %model.svm.solver = 'liblinear' ;
  model.svm.biasMultiplier = 1 ;

  if ~exist(conf.vocabFile) || conf.clobber
    vocab = trainVocabulary;
    disp('Vocabulary has been trained');
  else
    load(conf.vocabFile);
    disp('Vocabulary has been loaded');
  end

  model.vocab = vocab;
  
  if strcmp(model.quantizer, 'kdtree')
    model.kdtree = vl_kdtreebuild(model.vocab) ;
  end
  
  if ~exist(conf.histFile) || conf.clobber
    hists = computeSpatialHistorams;
    disp('Spatial histograms have been computed.');
  else
    load(conf.histFile);
    disp('Spatial histograms have been loaded.');
  end
 
  if ~exist(conf.histFile) || conf.clobber
    psix  = computeFeatureMap(hists);
    disp('Feature map has been computed');
  else
    load(conf.fmapFile);
    disp('Feature map has been loaded');
  end

  if ~exist(conf.modelFile) || conf.clobber
    trainSvm(psix);
    disp('SVM trained and model has been created');
  else 
    load(conf.modelFile);
    disp('SVM model has been loaded');
  end
  
  classifications = classifyImages(conf.trainDataPath, trainingData.imageFileNames(10000:11000));
  save(conf.classFile, 'classifications') ;
  disp('Images have been classified');
  
  %% classifier performance
  CP = classperf(trainingData.original.imageClasses, classifications)
  save(conf.cpFile, 'CP');                       
  disp('Classifier Performace has been calculated');
  
  %% create resultFile
  createResultFile(classifications);
  disp('Result file has been created');
end

function vocab = trainVocabulary 
  global conf trainingData model;
  descrs = {} ;
  parfor i=1:length(trainingData.imageClasses)
    imageFilePath = fullfile(conf.trainDataPath, sprintf('%d.png', i));
    im = imread(imageFilePath) ;
    im = standarizeImage(im) ;
    [drop, descrs{i}] = vl_phow(im, model.phowOpts{:}) ;
  end
  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;
  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, model.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
  save(conf.vocabFile, 'vocab') ;
end

function hists = computeSpatialHistorams
  global conf trainingData model;
  hists = {} ;
  parfor i=1:length(trainingData.imageClasses)
    imageFilePath = fullfile(conf.trainDataPath, sprintf('%d.png', i));
    im = imread(imageFilePath) ;
    hists{i} = getImageDescriptor(im);
  end
  hists = cat(2, hists{:}) ;
  save(conf.histFile, 'hists') ;
end

function psix = computeFeatureMap(hists) 
  global conf;
  psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;
  save(conf.fmapFile, 'psix') ;
end

function trainSvm(psix)
  global conf trainingData model;
  switch model.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (model.svm.C * length(trainingData.imageClasses)) ;
      w = [] ;
      %for ci = 1:length(labels)
      parfor ci = 1:length(trainingData.imageClasses)
        y = 2 * (trainingData.imageClassIds == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, trainingData.imageClassIds), y, lambda, ...
          'Solver', model.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', model.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end
    case 'liblinear'
      svm = train(trainingData.imageClassIds', ...
                  sparse(double(psix)),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                  model.svm.biasMultiplier, model.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end
  model.b = model.svm.biasMultiplier * b ;
  model.w = w ;
  save(conf.modelFile, 'model') ;
end

function im = standarizeImage(im)
  im = im2single(im) ;
  if size(im,1) > 480
    im = imresize(im, [480 NaN]) ;
  end
end

function hist = getImageDescriptor(im)
  global model;
  im = standarizeImage(im) ;
  width = size(im,2) ;
  height = size(im,1) ;
  numWords = size(model.vocab, 2) ;

  % get PHOW features
  [frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

  % quantize local descriptors into visual words
  switch model.quantizer
    case 'vq'
      [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
    case 'kdtree'
      binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
        single(descrs), ...
        'MaxComparisons', 50)) ;
  end

  for i = 1:length(model.numSpatialX)
    binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
    binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

    % combined quantization
    bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
      binsy,binsx,binsa) ;
    hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
    hist = vl_binsum(hist, ones(size(bins)), bins) ;
    hists{i} = single(hist / sum(hist)) ;
  end
  hist = cat(1,hists{:}) ;
  hist = hist / sum(hist) ;
end

function [className, score] = classify(im)
  global trainingData model;
  hist = getImageDescriptor(im) ;
  psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
  scores = model.w' * psix + model.b' ;
  [score, best] = max(scores) ;
  className = trainingData.imageClassSet{best} ;
end

function classifications = classifyImages(path, images)
  classifications = {};
  parfor i=1:length(images)
    fileName = sprintf('%d.%s',images(i),'png');
    im = imread(fullfile(path, fileName));
    [className, score] = classify(im);
    classifications{i} = className;
  end
end

function createResultFile(classifications)
  time = clock;
  outputfile = sprintf('team_bruteforce_submittal_%d-%d-%d-%d%d.txt', ...
    time(1), time(2), time(3), time(4), time(5));
  fileID = fopen(outputfile,'wt');
  fprintf(fileID,'id,label\n');
  for i=1:length(classifications)
    fprintf(fileID,'%d,%s\n\r',i,classifications{i});
  end
  fclose(fileID);
end

function data  = importData(filename, startRow, endRow)

  %% Initialize variables.
  delimiter = ',';
  if nargin<=2
    startRow = 2;
    endRow = inf;
  end

  %% Format string for each line of text:
  %   column1: double (%f)
  %	column2: text (%s)
  % For more information, see the TEXTSCAN documentation.
  formatSpec = '%f%s%[^\n\r]';

  %% Open the text file.
  fileID = fopen(filename,'r');

  %% Read columns of data according to format string.
  % This call is based on the structure of the file used to generate this
  % code. If an error occurs for a different file, try regenerating the code
  % from the Import Tool.
  dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
  for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
      dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
  end

  %% Close the text file.
  fclose(fileID);

  %% Save original image class order
  data.original.imageClasses = dataArray(:,2)';
  
  %% sort data array first accoding to labels and then filename
  dataArraySorted = sortrows([num2cell(dataArray{1}), dataArray{2}], [2 1]);

  %% names of the image files
  data.imageFileNames   = cell2mat(dataArraySorted(:,1))';

  %% assigned class of the image file
  data.imageClasses     = dataArraySorted(:, 2)';

  %% image class set  (in cifar there are 10)
  data.imageClassSet    = unique(data.imageClasses);

  %% assigned numeric class id for classes
  data.imageClassSetId  = 1:length(data.imageClassSet);

  %% class id for each image file
  data.imageClassIds    = [];

  %% populate data.labelids
  for i = 1:length(data.imageFileNames)
    for j = 1:length(data.imageClassSet)
      if strcmp(data.imageClasses{i}, data.imageClassSet{j})
        data.imageClassIds(i) = data.imageClassSetId(j);
        break
      end
    end
  end

end