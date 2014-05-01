function psix = compute_feature_map(hists) 
  global settings;
  psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;
  save(settings.file.fmap, 'psix') ;
  disp('Feature map has been computed');
end