function classify_nofold_all()

  settings = get_settings();
  
  data = import_traning_data(...
            fullfile(settings.train.dataPath, settings.train.file));
  
  model = get_initial_model();
          
  model = train_classifer(model, data);
  
  classify_competition_data(model);

end

