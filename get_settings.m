function settings = get_settings()
  settings.clobber         = true;
  settings.trainDataPath   = 'train';
  settings.testDataPath    = 'test';

  settings.trainDataSize   = 10000;
  settings.numOfFold       = 10;

  settings.vocabFile       = 'vocab.mat';
  settings.histFile        = 'hists.mat';
  settings.modelFile       = 'model.mat';
  settings.classFile       = 'class.mat';
  settings.dataFile        = 'trainingData.mat';
  settings.fmapFile        = 'fmap.mat';
  settings.cpFile          = 'cp.mat';
end