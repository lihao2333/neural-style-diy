require 'torch'
function preprocess(img)
	local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68}):view(3,1,1)
	img = img:mul(256.0) --选择一维通道中, 3,2,1通道
	img:add(-1, mean_pixel:expandAs(img)) -- 减去平均值
	return img
end

function deprocess(img)
	local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68}):view(3,1,1)
	img:add(1,mean_pixel:expandAs(img))
	img = img:div(256.0)
	return img
end

function setup_images(p)--get init_image, content_image, style_image
  	init_image = image.load(p.init_image, 3)
	init_image = image.scale(init_image, p.image_size, 'bilinear')
	p.init_image = preprocess(init_image):float()
  	content_image = image.load(p.content_image, 3)
	content_image = image.scale(content_image, p.image_size, 'bilinear')
	p.content_image = preprocess(content_image):float()
  	style_image = image.load(p.style_image, 3)
	style_image = image.scale(style_image, p.image_size, 'bilinear')
	p.style_image = preprocess(style_image):float()
end

function setup_device(p) --get dtype
  	if p.backend == 'nn' then
	  p.dtype = 'torch.FloatTensor'
 	else
	  require 'cltorch'
	  require 'clnn'
	  p.dtype = 'torch.ClTensor'
	  p.gpu = tonumber(p.gpu)
	  cltorch.setDevice(p.gpu)
	end
end

