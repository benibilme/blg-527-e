classdef Classifier
  
  properties
    model
  end
  
  methods
    function obj = Classifier(classSet, model)
      obj.classSet = classSet;
      obj.model    = model;
    end
    
    function [className, classId] = classify_image(im)
      hist = get_image_descriptor(im);
      psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5);
      scores = model.w' * psix + model.b';
      [score, best] = max(scores);
      classId   = best;
      className = classSet{classId};
    end
  end
  
  
end

