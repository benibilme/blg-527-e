function data  = importfile2(filename, startRow, endRow)

    %% Initialize variables.
    delimiter = ',';
    if nargin<=2
        startRow = 2;
        endRow = inf;
    end

    %% Format string for each line of text:
    %   column1: double (%f)
    %	column2: text (%s)
    % For more information, see the TEXTSCAN documentation.
    formatSpec = '%f%s%[^\n\r]';

    %% Open the text file.
    fileID = fopen(filename,'r');

    %% Read columns of data according to format string.
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

    %% Close the text file.
    fclose(fileID);

    %% Allocate imported array to column variable names

    %% sort data array first accoding to labels and then filename
    dataArraySorted = sortrows([num2cell(dataArray{1}), dataArray{2}], [2 1])

    %% names of the image files 
    data.filenames   = cell2mat(dataArraySorted(:,1))';

    %% assigned label of the image file
    data.labels      = dataArraySorted(:, 2)';

    %% number of classes (in cifar there are 10)
    data.labelset    = unique(data.labels);

    %% assigned numeric class id for classes 
    data.labelsetid  = 1:length(data.labelset);

    %% class id for each image file  
    data.labelids    = [];

    %% populate data.labelids
    for i = 1:length(data.filenames)
        for j = 1:length(data.labelset)
            if strcmp(data.labels{i}, data.labelset{j})
                data.labelids(i) = data.labelsetid(j);
                break
            end
        end
    end
    
end




