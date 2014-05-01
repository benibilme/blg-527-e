function classify_all_nofold(trainDataSize)

  global settings;
  
  settings = get_settings();
  model    = get_initial_model();
  
  data = import_traning_data(...
            fullfile(settings.train.dataPath, settings.train.file));
  
  selectedIndices = randperm(length(data.imageClasses), ... 
                             trainDataSize);
                           
  data = get_subset_data(data, selectedIndices);
          
  model = train_classifer(model, data);
  
  classify_competition_data(model);

end

