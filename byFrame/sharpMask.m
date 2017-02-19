function status =  sharpMask(v, img_path, res_path, sm_path, n, motion_path)
status = -1; 
if exist(img_path, 'file') == 2; 
  if v == 1
    status = system(['th '  sm_path  'computeMasks.lua -model ' sm_path '/pretrained/sharpmask/  -res ' res_path ' -v 1 -img ' img_path  ' -objs ' n]); 
  elseif v == 2
    status = system(['th '  sm_path  'computeMasks.lua -model ' sm_path '/pretrained/sharpmask/  -res ' res_path ' -v 2 -img ' img_path  ' -mask ' motion_path]); 
  end
else
  fprintf('Image file does not exist..'); 

if status ~= 0
  fprintf('Sharpmask did not succeed in creating masks'); 
end

end
