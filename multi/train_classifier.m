function model = train_classifier(data)

  global settings model;

  vocab = train_vocabulary(...
              model, ...
              settings.train.dataPath, ...
              data.imageFileNames);
            
  model.vocab = vocab;
  
  if strcmp(model.quantizer, 'kdtree')
    model.kdtree = vl_kdtreebuild(model.vocab) ;
  end
  
  hist  = compute_spatial_histograms(...
             settings.train.dataPath, ...
             data.imageFileNames);   
           
  psix  = compute_feature_map(hists);
  
  model = train_svm(model, data, psix);
  
  disp('Classifier is trained');
end