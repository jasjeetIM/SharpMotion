--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Run full scene inference in sample image
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'
--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-img','data/testImage.jpg' ,'path/to/test/image')
cmd:option('-gpu', 2, 'gpu device')
cmd:option('-np', 100,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')
cmd:option('-pdir', '/data2/jdhaliwal/Sharpmask/deepmask/pdir/', 'parent dir containing folders for each video') 
cmd:option('-sdir', '/data2/jdhaliwal/Sharpmask/deepmask/sdir/', 'directory to save results')
cmd:option('-m', 1, 'motionMasks or imgMasks')
cmd:option('jpg_masks',0,'motionMasks saved as images')
local config = cmd:parse(arg)

--[[ pdir:

1) pdir is the directory that contains the jpeg frames to be segmented as well as the motion frames. 
2) The structure of pdir is as follows: 

pdir/
    --> [video1_name]/
                     --> {frame_1, ..., frame_n}

                      --> [video1_name]_motion/
                                              --> {mframe_1, ..., mframe_n}
            .
            .
            .
  
 
3) 'frame_i' and 'mframei' should be named the same for a video. 
                           
4) The pdir parameter should contain the trailing slash.
 Eg: If the parent directory is the ~ directory, then set pdir = '/home/'


sdir:

1) sdir is the directory where the algorithm will store the masks.  
2) The structure of sdir is as follows:

sdir/
    --> [video1_name]/
                     --> 'sframe_i'/
                                  --> {mask_1, ..., mask_n}
                                  -->scores.cvs

2) The name of "sframe_i" will be the same as the name of "frame_i" 
   for a video but wont include the '.jpg' extension.  
3) 'mask_i' will be named: 'sframe_i_mask[j].csv' where j = 1, ..,
   If you set "jpg_masks' to 1, then we will also save 'sframe_i_mask[j].jpg' for j = 1,..., 
3) scores.csv contain comma seperated values such that each field will contain 
   the string "'sframe_i_mask[j].csv':score(mask_j)"
4) The sdir parameter should contain the trailing slash. 
   Eg: If sdir is ~, then set sdir = '/home/'
5) sdir should be an empty directory in the first iteration. 
This means, when you first run the code, sdir should be empty. 
The program will automatically write the results to it. If you run the program multiple 
times, it will simply overwrite the existing results. 


 'm': if set to 1, we will save motion masks, otherwise we will store image masks 
      in a 'segmentation' subdirectory
 'jpg_masks': set to 1 if you want to save masks as images for visualization
]]
--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local coco = require 'coco'
local maskApi = coco.MaskApi

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load moodel
paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

print('| loading model file... ' .. config.model)
local m = torch.load(config.model..'/model.t7')
local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.si,config.sf,config.ss do 
  table.insert(scales,2^i) 
end

if torch.type(model)=='nn.DeepMask' then
  paths.dofile('InferDeepMask.lua')
elseif torch.type(model)=='nn.SharpMask' then
  paths.dofile('InferSharpMask.lua')
end

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = config.dm,
}

--------------------------------------------------------------------------------
-- do it
print('| start')
local csv = require('csvigo')
local pdir_path = config.pdir
local sdir_path = config.sdir
local t, popen = {}, io.popen
local test, err = io.open(pdir_path)
if err~=nil then 
  if string.match(err, 'No such file or directory') then
    print ("Error: 'pdir' directory does not exit") 
    os.exit()
  else print("Error: 'pdir' directory returned a nil value")
  os.exit()
  end
end    

test, err = io.open(sdir_path)
if err~=nil then 
  if string.match(err, 'No such file or directory') then
    print ("Error: 'sdir' directory does not exit") 
    os.exit()
  else print("Error: 'sdir' directory returned a nil value")
  os.exit()
  end
end    

local pdir = popen('ls '..pdir_path)
local sdir = popen('ls '..sdir_path)
local motion = config.m
local jpg_masks = config.jpg_masks
--Loop over all the folders in the directory
for dirname in pdir:lines() do
  local pfolder = popen('ls '..pdir_path..dirname)
  --Find the motion folder
  local pmotion, err = io.open(pdir_path .. dirname..'/'..dirname..'_motion/')
  --Error in locating motion folder, lets move to the next video. 
  if err~= nil then 
    if string.match(err, 'No such file or directory') then 
    print ("Error : no motion directory found for video: ", dirname) 
    print ("Moving onto next video...")
    end
  -- No error in locating motion folder, lets segment. 
  else
    io.close(pmotion)
    -- Segment frame by frame
    for filename in pfolder:lines() do  
        if filename ~= "GroundTruth" and filename ~= dirname..'_motion' then
          t = os.time()
          filename_dir = filename:sub(1,filename:len()-4)
          -- load image
          local img = image.load(pdir_path .. dirname ..'/'.. filename)
          local h,w = img:size(2),img:size(3)
          -- forward all scales
          infer:forward(img)

          -- get top propsals
          local motion_file = pdir_path ..dirname..'/'..dirname..'_motion/'.. filename
          local masks,topscores = infer:getTopProps(motion,.2, h, w, filename,motion_file)


          -- save result and write scores. 
          -- We need to create required directories
          -- We create a new scores.csv file every time 
          local v,err = io.open(sdir_path .. dirname)
          if err~= nil then if string.match(err, "No such file or directory")  then os.execute('mkdir '..sdir_path .. dirname) end end
	  local f, err = io.open(sdir_path..dirname..'/'..filename_dir)
          if err~= nil then if string.match(err, "No such file or directory") then os.execute('mkdir ' ..sdir_path ..dirname ..'/'.. filename_dir) end end
          local scores, err = io.open(sdir_path ..dirname ..'/' ..filename_dir..'/' .. 'scores.csv' , "r")
          if err ~= nil then 
              if string.match(err, "No such file or directory") then 
                scores = io.open(sdir_path ..dirname ..'/' ..filename_dir..'/' .. 'scores.csv' , "a")
              end
          else 
             os.execute('rm ' .. sdir_path .. dirname ..'/'.. filename_dir ..'/' .. 'scores.csv')
             scores = io.open(sdir_path ..dirname ..'/' ..filename_dir..'/' .. 'scores.csv' , "a")
          end

          -- Check to see if we have alteast 1 mask and 1 score. If so, proceed to write results. 
          if topscores:sum() > 0 then
          for j=1,torch.nonzero(topscores):size(1) do
            local t2 = {}
	    for x=1,masks[j]:size(1) do
  	        t2[x] = {}
     		for y=1,masks[j]:size(2) do
    			t2[x][y] = masks[j][x][y]
  		end
	    end
            csv.save({path = string.format(sdir_path..dirname ..'/'..filename_dir..'/'..filename_dir .. '_mask'.. j ..'.csv'), data = t2, verbose=false})
            -- Save masks as jpegs for visualization
            if jpg_masks == 1 then 
              image.save(string.format(sdir_path..dirname ..'/'..filename_dir..'/'..filename_dir .. '_mask'.. j ..'.jpg'),masks[j])
	    end
            scores:write(filename_dir..'_mask'..j..'.csv'..':'..topscores[j]..',')
        end end
         io.write(string.format("Masks for frame %s saved in ~ %s seconds\n",filename,os.time() - t))
	
        -- If motion ==0, then we can save the masks on the image and write the image to disk. 
        if motion == 0 then
          local seg, err = io.open(sdir_path ..dirname ..'/' ..'segmentations/')
          if err ~= nil then 
            if string.match(err, "No such file or directory") then os.execute('mkdir ' .. sdir_path ..dirname ..'/'.. 'segmentations/') end end
          local res = img:clone()
	  maskApi.drawMasks(res, masks, 10)
	  image.save(string.format(sdir_path..dirname ..'/'..'segmentations/'..filename),res)
        end
       end
      end
      pfolder:close()
    end 
end
  
print('| done')
pdir:close()
sdir:close()
collectgarbage()
