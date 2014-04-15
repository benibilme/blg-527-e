% YAPILACAKLAR
% TEN FOLD CROSSVALIDATION GORE KOD DUZENELENECEK
% CLASSIFIER PERFORMANCE PARAMETRELERI CIKARILACAK ROUTINE TAMAMLANACAK
% 



function classifier()

conf.clobber         = false; 
conf.trainDataPath   = 'train';
conf.testDataPath    = 'test';
conf.vocabFile       = 'vocab.mat';
conf.histFile        = 'hists.mat';
conf.modelFile       = 'model.mat';
conf.resultFile      = 'result' ;

randn('state', 1) ;
rand('state', 1) ;
vl_twister('state', 1) ;

% --------------------------------------------------------------------
%  Setup data
% --------------------------------------------------------------------

[trainingIds, trainingLabels] =  importfile('trainLabels.csv',2, 1000);

labels = unique(trainingLabels);
labelVals = 1:length(labels);

disp('Data setup is completed');

% --------------------------------------------------------------------
%  Setup up model parameters 
% --------------------------------------------------------------------

%%CLASSES için bir şey yapılacak
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
model.classes  = labels;
model.classify = @classify ;

disp('Model setup is completed');




% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------

if ~exist(conf.vocabFile) || conf.clobber 

  descrs = {} ;
  parfor i=1:length(trainingIds)
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
else
  load(conf.vocabFile) ;
end

model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end

disp('Vocabulary has been trained');

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------

% NOT : Burada test mi train mi kullanilmasi gerektigini anlamadim
% Simdilik train data referans aldim.
if ~exist(conf.histFile) || conf.clobber 
  hists = {} ;
  parfor i=1:length(trainingIds)
    imageFilePath = fullfile(conf.trainDataPath, sprintf('%d.png', i));
    im = imread(imageFilePath) ;
    hists{i} = getImageDescriptor(model, im);
  end

  hists = cat(2, hists{:}) ;
  save(conf.histFile, 'hists') ;
else
  load(conf.histFile) ;
end

disp('Spatial histograms have been computed.');
% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------

psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;

disp('Feature map has been computed');

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

if ~exist(conf.modelFile) || conf.clobber
  switch model.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (model.svm.C * length(trainingIds)) ;
      w = [] ;
      for ci = 1:length(labels)
      %parfor ci = 1:length(labels)
        y = 2 * (trainingIds == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, trainingIds), y, lambda, ...
          'Solver', model.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', model.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(trainingIds', ...
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
else
  load(conf.modelFile) ;
end

disp('SVM trained and model has been created');

% --------------------------------------------------------------------
%  classify test data and generate submittal file
% --------------------------------------------------------------------
time = clock;
outputfile = sprintf('team_bruteforce_submittal_%d-%d-%d-%d%d.txt', ...
                     time(1), time(2), time(3), time(4), time(5));
testimages = dir(fullfile(conf.testDataPath,'*.png')); 
fileID = fopen(outputfile,'wt');
fprintf(fileID,'id,label\n');
for i=1:length(testimages)
   im = imread(fullfile(conf.testDataPath, testimages(i).name));
   [classification, score] = classify(model, im);
   fprintf(fileID,'%d,%s\n',i,classification{1});
end
fclose(fileID);

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------

% Estimate the class of the test images

% scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
% [drop, imageEstClass] = max(scores, [], 1) ;

% Compute the confusion matrix

% idx = sub2ind([length(trainingSetLabels), length(trainingSetLabels)], ...
%               trainingIds, imageEstClass) ;
% confus = zeros(length(trainingSetLabels)) ;
% confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
% figure(1) ; clf;
% subplot(1,2,1) ;
% imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
% set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
% subplot(1,2,2) ;
% imagesc(confus) ;
% title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
%               100 * mean(diag(confus)/conf.numTest) )) ;
% print('-depsc2', [conf.resultPath '.ps']) ;
% save([conf.resultPath '.mat'], 'confus', 'conf') ;

% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end

% -------------------------------------------------------------------------
function hist = getImageDescriptor(model, im)
% -------------------------------------------------------------------------

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

% -------------------------------------------------------------------------
function [className, score] = classify(model, im)
% -------------------------------------------------------------------------

hist = getImageDescriptor(model, im) ;
psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes(best) ;