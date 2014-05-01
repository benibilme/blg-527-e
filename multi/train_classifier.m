function model = train_classifier(model, data)

  global settings;

  vocab = train_vocabulary(...
              settings.train.dataPath, ...
              data.imageFileNames, ...
              model.phowOpts, ...
              model.numWords);
            
  model.vocab = vocab;
            
  if strcmp(model.quantizer, 'kdtree')
    model.kdtree = vl_kdtreebuild(model.vocab) ;
  end
  
  hists  = compute_spatial_histograms(...
             model, ...
             settings.train.dataPath, ...
             data.imageFileNames);   
           
  psix  = compute_feature_map(hists);
  
  model = train_svm(model, data, psix);
  
  disp('Classifier is trained');
end