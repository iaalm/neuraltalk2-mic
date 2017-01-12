local utils = require 'misc.utils'
local net_utils = {}
local fb_transforms = require 'misc.transforms'

-- Load resnet from facebook
function net_utils.build_cnn(cnn, opt)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  local cnn_remove = utils.getopt(opt, 'cnn_remove', 1)
  local cnn_outsize = utils.getopt(opt, 'cnn_outsize', 2048)
  
  local cnn_part = cnn
  for _=1,cnn_remove do
    cnn_part:remove(#cnn_part.modules)
  end

  cnn_part:add(nn.Linear(cnn_outsize,encoding_size))
  cnn_part:add(nn.ReLU(true))
  
  if backend == 'cudnn' then
    cudnn.convert(cnn_part, cudnn)
  end
  return cnn_part
end

-- takes a batch of images and preprocesses them
-- imgs need to be 0..1, and 224 as size to forward
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')
  
  imgs = imgs:float() 
  imgs:div(255)
  local meanstd = {
     mean = { 0.485, 0.456, 0.406 },
     std = { 0.229, 0.224, 0.225 },
  }

  local transform = fb_transforms.Compose{
     fb_transforms.ColorNormalize(meanstd),
     fb_transforms.CenterCrop(224),
  }
  
  local cnn_input_size = 224
  imgs_out = torch.Tensor(imgs:size(1), imgs:size(2), cnn_input_size, cnn_input_size):type(imgs:type())
  for i = 1, imgs:size(1) do
    imgs_out[i] = transform(imgs[i])
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs_out = imgs_out:cuda() else imgs_out = imgs_out:float() end

  return imgs_out
end

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, ground_truth, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json ' .. ground_truth) -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  os.remove('coco-caption/val' .. id .. '.json_out.json')
  os.remove('coco-caption/val' .. id .. '.json')
  return result_struct
end

return net_utils
