function classify_competition_data(model)
   
  global settings;
  
  settings = get_settings();
  
  if nargin ~= 1
    load(settings.file.model);
  end 
   
  disp('Classification of competition data begins...')
  classifications = classify_images(model, ...
                                    settings.testDataPath, ...
                                    [1:300000]);
  save('cifar10classes.mat', 'classifications');
                                 
  disp('Competition data result file is being created')
  create_results_file(classifications);
end

