require 'torch'
require 'image'
require 'loadcaffe'
require 'tools'
require 'tools_network'
require 'tools_train'
local cmd = torch.CmdLine()
cmd:option('-content_image','contents/me.jpg','content image')
cmd:option('-style_image','styles/picasso.jpg','style image')
cmd:option('-image_size','512', 'image size')
--cmd:option('-backend', 'clnn', 'nn|clnn')
cmd:option('-backend', 'nn', 'nn|clnn')
cmd:option('-loadcaffe_backend', 'nn', 'nn')
cmd:option('-gpu', '1', 'gpu id')
cmd:option('-init_image', 'contents/me.jpg', 'init_image')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-normalize_gradients', 'false')
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 20)
cmd:option('-content_layers', 'relu4_2')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
cmd:option('-seed', '1234')
cmd:option('-learning_rate', 1e1)
cmd:option('-num_iterations', 1000)
cmd:option('-print_iter', 1)
cmd:option('-save_iter', 1)

cmd:option('-output_image', 'output/out.png')

local function main(p)
    setup_device(p)-- get dtype
	setup_images(p)-- get init_image, content_image, style_image
	setup_network_1(p)
	setup_network_2(p)
	train(p)
end
local p = cmd:parse(arg)
p.output_image = string.format("%s/res.%s",paths.dirname(p.content_image),paths.extname(p.content_image))
main(p)
