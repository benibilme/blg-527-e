function model = get_initial_model()
  model.numWords = 600;
  model.phowOpts = {'Step', 3};
  model.numSpatialX = [2 4];
  model.numSpatialY = [2 4];
  model.quantizer = 'vq';
  model.vocab = [] ;
  model.w = [] ;
  model.b = [] ;
  model.svm.C = 10 ;
  model.svm.solver = 'sdca' ;
  %model.svm.solver = 'sgd' ;
  %model.svm.solver = 'liblinear' ;
  model.svm.biasMultiplier = 1 ;
end