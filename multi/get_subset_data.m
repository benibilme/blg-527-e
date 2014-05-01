function subsetData = get_subset_data(data, indices)
  subsetData.imageFileNames  = data.imageFileNames(indices);
  subsetData.imageClasses    = data.imageClasses(indices);
  subsetData.imageClassIds   = data.imageClassIds(indices);
  subsetData.imageClassSet   = data.imageClassSet;
  subsetData.imageClassSetId = data.imageClassSetId;
end

