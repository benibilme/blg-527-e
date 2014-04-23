function create_result_file(classifications)
  time = clock;
  outputfile = sprintf('team_bruteforce_submittal_%d-%d-%d-%d%d.txt', ...
    time(1), time(2), time(3), time(4), time(5));
  fileID = fopen(outputfile,'wt');
  fprintf(fileID,'id,label\n');
  for i=1:length(classifications)
    fprintf(fileID,'%d,%s\n',i,classifications{i});
  end
  fclose(fileID);
end