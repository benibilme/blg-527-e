function [className, classId] = classify_image(model, classSetLabels, im)
  hist = get_image_descriptor(model, im);
  psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5);
  scores = model.w' * psix + model.b';
  [score, best] = max(scores);
  classId   = best;
  className = classSetLabels{classId};
end