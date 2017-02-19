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
cmd:option('-gpu', 2, 'gpu device')
cmd:option('-np', 100,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')
cmd:option('-img','path to the image frame') 
cmd:option('-mask', 'path to the motion frame')
cmd:option('-objs', 'number of proposals to return')
cmd:option('-res', 'directory where to write the masks')
cmd:option('-v', 'version 1 or version 2')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local img_path = config.img
local version = config.v
if version == 2 then 
  local mask_path = config.mask
end


local objects = config.objs
local res_path = config.res
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
-- load image
local csv = require('csvigo')
local img = image.load(img_path)
local h,w = img:size(2),img:size(3)
-- forward all scales
infer:forward(img)
-- get top propsals
if version == '1' then
  masks,topscores = infer:getTopProps1(.2, h, w, filename,objects)
else
  masks, topscores = infer:getTopProps2(.2, h, w, filename,mask_path )
end

local scores, err = io.open(res_path .. 'scores.csv' , "r")
if err ~= nil then 
  if string.match(err, "No such file or directory") then 
    scores = io.open(res_path .. 'scores.csv' , "a")
  end
else 
 os.execute('rm ' .. res_path  .. 'scores.csv')
 scores = io.open(res_path  .. 'scores.csv' , "a")
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
    csv.save({path = string.format(res_path.. 'mask'.. j ..'.csv'), data = t2, verbose=false})
    scores:write('mask '.. j .. ':'..topscores[j]..',')
    end 
end	
  
print('| done')
collectgarbage()
