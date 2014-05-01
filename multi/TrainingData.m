classdef TrainingData
  %UNTITLED10 Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    dataFile    = 'trainLabels.csv';
    fileNames   = {};
    classes     = {};
    classIds    = [];
    classSet    = [];
    classSetIds = [];
  end
  
  methods
    
    function obj = TrainingData(filename)
      
      obj.dataFile = filename;
       
      formatSpec = '%f%s%[^\n\r]';
      
      delimiter = ',';
      startRow = 2;
      endRow   = inf;
      
      % Open the text file.
      fileID = fopen(filename,'r');

      dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
      for block=2:length(startRow)
        frewind(fileID);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
        for col=1:length(dataArray)
          dataArray{col} = [dataArray{col};dataArrayBlock{col}];
        end
      end
      
      % Close the text file.
      fclose(fileID);
      
       % names of the image files
      imageFileNames = dataArray{1};
      
      for i=1:length(imageFileNames)
        obj.fileNames{i} = sprintf('%d.png', imageFileNames(i));
      end
      
      % assigned class of the image file
      obj.classes     = dataArray{2}';
      
      % image class set  (in cifar there are 10)
      obj.classSet    = unique(obj.classes);
      
      % assigned numeric class id for classes
      obj.classSetIds = 1:length(obj.classSet);
      
      % class id for each image file
      obj.classIds    = [];
      
      % populate data.labelids
      for i = 1:length(obj.fileNames)
        for j = 1:length(obj.classSet)
          if strcmp(obj.classes{i}, obj.classSet{j})
            obj.classIds(i) = obj.classSetIds(j);
            break
          end
        end
      end
    end
  end
end

