function settings = get_settings()
  settings.clobber         = true;

  settings.test.dataPath   = '../test';
  
  settings.train.file      = '../trainLabels.csv';
  settings.train.dataPath  = '../train';

  settings.file.vocab       = 'vocab.mat';
  settings.file.hist        = 'hists.mat';
  settings.file.model       = 'model.mat';
  settings.file.finalmodel  = 'finalmodel.mat';
  settings.file.classes     = 'class.mat';
  settings.file.data        = 'tdata.mat';
  settings.file.fmap        = 'fmap.mat';
  settings.file.cp          = 'cp.mat';
  settings.file.competition = 'cifar10classes.mat';
end