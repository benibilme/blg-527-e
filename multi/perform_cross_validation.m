function model = perform_cross_validation(data, trainDataSize, numOfFold)
  
  global settings;

  disp('...Cross Validation Begins...')
  
  settings = get_settings();
  model    = get_initial_model();
 
  selectedIndices = randperm(length(data.imageClasses), ... 
                             trainDataSize);

  selectedData = get_subset_data(data, selectedIndices);

  %Perform Ten-Fold Cross Validation
  indices = crossvalind('Kfold', ...
                        selectedData.imageClasses, ...
                        numOfFold);
  
  cp = classperf(selectedData.imageClasses);
  
  for i=1:numOfFold
    
    testIndices  = (indices == i);
    trainIndices = ~testIndices;
    
    testData     = get_subset_data(selectedData, testIndices);
    trainingData = get_subset_data(selectedData, trainIndices);
     
    model = train_classifier(model, trainingData);
  
    if (i > 1) 
      wTotal = wTotal + model.w;
      bTotal = bTotal + model.b;
    else   
      wTotal = model.w;
      bTotal = model.b;
    end
    
    disp('Classifying test data')
    classifications = classify_images(model, ...
                                      testData, ...
                                      settings.train.dataPath);
    % Calculate Classifier Performance
    disp('Classifier Performace for this iteration')
    cp = classperf(testData.imageClasses, classifications)
    save('cp.mat', 'cp');
  end
  
  disp('Overall Classifier Performance')
  cp
  
  % average w and d
  model.w = wTotal ./ numOfFold;
  model.b = bTotal ./ numOfFold;
  save('modelFinal.mat', 'model');
  
  disp('...Cross Validation Ends...')
end

