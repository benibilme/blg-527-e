function data  = import_training_data(filename, startRow, endRow)

  % Initialize variables.
  delimiter = ',';
  if nargin<=2
    startRow = 2;
    endRow = inf;
  end

  % Format string for each line of text:
  %   column1: double (%f)
  %	column2: text (%s)
  % For more information, see the TEXTSCAN documentation.
  formatSpec = '%f%s%[^\n\r]';

  % Open the text file.
  fileID = fopen(filename,'r');

  % Read columns of data according to format string.
  % This call is based on the structure of the file used to generate this
  % code. If an error occurs for a different file, try regenerating the code
  % from the Import Tool.
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
  data.imageFileNames   = dataArray{1};

  % assigned class of the image file
  data.imageClasses     = dataArray{2};

  % image class set  (in cifar there are 10)
  data.imageClassSet    = unique(data.imageClasses);

  % assigned numeric class id for classes
  data.imageClassSetId  = 1:length(data.imageClassSet);

  % class id for each image file
  data.imageClassIds    = [];

  % populate data.labelids
  for i = 1:length(data.imageFileNames)
    for j = 1:length(data.imageClassSet)
      if strcmp(data.imageClasses{i}, data.imageClassSet{j})
        data.imageClassIds(i) = data.imageClassSetId(j);
        break
      end
    end
  end
end