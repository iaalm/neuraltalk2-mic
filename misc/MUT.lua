require 'nn'
require 'nngraph'

local MUT = {}
function MUT.mut1(input_size, output_size, rnn_size, n, dropout_l, dropout_t, res_rnn)
  dropout_l = dropout_l or 0 
  dropout_t = dropout_t or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    if dropout_t > 0 then prev_h = nn.Dropout(dropout_t)(prev_h):annotate{name='drop_t_' .. L} end -- apply dropout_t, if any
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      if res_rnn > 0 and L > res_rnn + 1 and (L - 2) % res_rnn == 0 then    
        x = nn.CAddTable()({outputs[(L-1)*2], outputs[(L-1-res_rnn)*2]})    
      else
        x = outputs[(L-1)*2] 
      end
      if dropout_l > 0 then x = nn.Dropout(dropout_l)(x):annotate{name='drop_l_' .. L} end -- apply dropout_l, if any
      input_size_L = rnn_size
    end

    -- begin core unit
    local z = x - nn.Linear(input_size_L, rnn_size) - nn.Sigmoid()
    local nz = z - nn.MulConstant(-1) - nn.AddConstant(1,true)
    local rx = x - nn.Linear(input_size_L, rnn_size)
    local rr = prev_h - nn.Linear(rnn_size, rnn_size)
    local r = nn.CAddTable()({rx, rr}) - nn.Sigmoid()
    local tx = x - nn.Tanh()
    local hr = nn.CMulTable()({prev_h, r}) - nn.Linear(rnn_size, rnn_size)
    local m = nn.CAddTable()({hr, tx}) - nn.Tanh()
    local next_h           = nn.CAddTable()({
        nn.CMulTable()({m,      z}),
        nn.CMulTable()({prev_h, nz})
      })
    -- end core unit

    table.insert(outputs, prev_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout_l > 0 then top_h = nn.Dropout(dropout_l)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end
function MUT.mut3(input_size, output_size, rnn_size, n, dropout_l, res_rnn)
  dropout_l = dropout_l or 0 
  dropout_t = dropout_t or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    if dropout_t > 0 then prev_h = nn.Dropout(dropout_t)(prev_h):annotate{name='drop_t_' .. L} end -- apply dropout_t, if any
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      if res_rnn > 0 and L > res_rnn + 1 and (L - 2) % res_rnn == 0 then    
        x = nn.CAddTable()({outputs[(L-1)*2], outputs[(L-1-res_rnn)*2]})    
      else
        x = outputs[(L-1)*2] 
      end
      if dropout_l > 0 then x = nn.Dropout(dropout_l)(x):annotate{name='drop_l_' .. L} end -- apply dropout_l, if any
      input_size_L = rnn_size
    end

    -- begin core unit
    local zx = x - nn.Linear(input_size_L, rnn_size)
    local zh = prev_h - nn.Tanh() - nn.Linear(rnn_size, rnn_size)
    local z = nn.CAddTable()({zx, zh}) - nn.Sigmoid()
    local nz = z - nn.MulConstant(-1) - nn.AddConstant(1,true)
    local rx = x - nn.Linear(input_size_L, rnn_size)
    local rr = prev_h - nn.Linear(input_size_L, rnn_size)
    local r = nn.CAddTable()({rx, rr}) - nn.Sigmoid()
    local tx = x - nn.Linear(input_size_L, rnn_size)
    local hr = nn.CMulTable()({prev_h, r}) - nn.Linear(rnn_size, rnn_size)
    local m = nn.CAddTable()({hr, tx}) - nn.Tanh()
    local next_h           = nn.CAddTable()({
        nn.CMulTable()({m,      z}),
        nn.CMulTable()({prev_h, nz})
      })
    -- end core unit

    table.insert(outputs, prev_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout_l > 0 then top_h = nn.Dropout(dropout_l)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return MUT

