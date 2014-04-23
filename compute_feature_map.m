function psix = compute_feature_map(hists) 
  psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;
  %save(conf.fmapFile, 'psix') ;
end