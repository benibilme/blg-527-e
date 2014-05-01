function im = get_standart_image(path)
  im = im2single(im) ;
  if size(im,1) > 480
    im = imresize(im, [480 NaN]) ;
  end
end