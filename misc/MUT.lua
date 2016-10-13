require 'nn'
require 'nngraph'

local MUT = {}
function MUT.mut1(input_size, output_size, rnn_size, n, dropout, res_rnn)
  dropout = dropout or 0 

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
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- begin core unit
    local z = x - nn.Linear(input_size_L, rnn_size) - nn.Sigmoid()
    local nz = z - nn.Mul(-1) - nn.AddConstant(1,true)
    local rx = x - nn.Linear(input_size_L, rnn_size)
    local rr = prev_h - nn.Linear(input_size_L, rnn_size)
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
    if res_rnn > 0 and L > res_rnn and (L - 1) % res_rnn == 0 then
      table.insert(outputs, nn.CAddTable()({next_h, outputs[(L-res_rnn)*2]}))
    else
      table.insert(outputs, next_h)
    end 
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return MUT

