function [className, classId] = classify_image(classLabels, model, im)
  hist = get_image_descriptor(im);
  psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5);
  scores = model.w' * psix + model.b';
  [score, best] = max(scores);
  classId   = best;
  className = classLabels{classId};
end