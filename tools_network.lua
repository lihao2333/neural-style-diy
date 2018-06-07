require 'optim'
require 'tools_loss'	

function setup_network_1(p)
local cnn = loadcaffe.load(p.proto_file, p.model_file, p.loadcaffe_backend):type(p.dtype)
print(cnn)

local content_losses, style_losses = {}, {}
local next_content_idx, next_style_idx = 1, 1
local net = nn.Sequential()

local content_layers = p.content_layers:split(",")
local style_layers = p.style_layers:split(",")



for i = 1, #cnn do
  if next_content_idx <= #content_layers or next_style_idx <=#style_layers then
	local layer = cnn:get(i)
	local name = layer.name
	local layer_type = torch.type(layer)
	net:add(layer)
    if name == content_layers[next_content_idx] then
	  local norm = p.normalize_gradients
	  local loss_module = nn.ContentLoss(p.content_weight, norm):type(p.dtype)
      net:add(loss_module)
      table.insert(content_losses, loss_module)
      next_content_idx = next_content_idx + 1
    end
    if name == style_layers[next_style_idx] then
	  local norm = p.normalize_gradients
	  local loss_module = nn.StyleLoss(p.style_weight, norm):type(p.dtype)
      net:add(loss_module)
      table.insert(style_losses, loss_module)
      next_style_idx = next_style_idx + 1
    end
  end
end
net:type(p.dtype)
print(net)
print(content_losses)
print(style_losses)
local net_box = {net, content_losses, style_losses}
torch.save('net_box.t7', net_box)



end

function setup_network_2(p) --capture content and style target

  print('capture')

  --net_box = torch.load('net_box_row.t7')
  net_box = torch.load('net_box.t7')
  p.net = net_box[1]
  p.content_losses = net_box[2]
  p.style_losses = net_box[3]
--  p.net = torch.load('net-adam.t8')
--  p.content_losses = torch.load('content-losses.t7')
--  p.style_losses = torch.load('style-losses.t7')
  print('init')
  print(p.content_losses)
  print(p.style_losses)

  for i=1,#p.style_losses do
	print(p.style_losses[i].mode)
  end
  for i=1,#p.content_losses do
	print("content number" .. i)
	p.content_losses[i].mode='capture'
	p.net:forward(p.content_image:type(p.dtype))
  end
  for i=1,#p.content_losses do
	print("content number" .. i)
	p.content_losses[i].mode = 'none'
  end
  for i=1,#p.style_losses do
	print("style number" .. i)
	p.style_losses[i].mode='capture'
	p.net:forward(p.style_image:type(p.dtype))
  end 

  print('captured')
  print(p.content_losses)
  print(p.style_losses)

  for i=1,#p.content_losses do
	p.content_losses[i].mode = 'loss'
  end
  for i=1,#p.style_losses do
	p.style_losses[i].mode = 'loss'
  end
  print('lossed')
  print(p.content_losses)
  print(p.style_losses)
  local net_box_lossed= {p.net, p.content_losses, p.style_losses}
  torch.save('net_box_lossed.t7', net_box_lossed)
  print('end')

end

