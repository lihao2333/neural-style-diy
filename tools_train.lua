require 'optim'
require 'torch'

function train(p)
  local net_box_lossed = torch.load('net_box_lossed.t7')
  p.net = net_box_lossed[1]
  p.content_losses =net_box_lossed[2]
  p.style_losses = net_box_lossed[3]
  print('run')
  print(p.net)
  print(p.content_losses)
  print(p.style_losses)
  local y = p.net:forward(p.init_image)
  local dy = p.init_image.new(#y):zero()
  local optim_state = {learningRate = p.learning_rate}
  local num_calls = 0

  local function build_filename(output_image, iteration)
    local ext = paths.extname(output_image)
    local basename = paths.basename(output_image, ext)
    local directory = paths.dirname(output_image)
    return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
  end
  local function maybe_save(t)
    local should_save = p.save_iter >= 1 and t % p.save_iter == 0
    should_save = should_save or t == p.num_iterations
    if should_save then
      local disp = deprocess(p.init_image:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(p.output_image, t)
      if t == p.num_iterations then
        filename = p.output_image
      end

      -- Maybe perform postprocessing for color-independent style transfer
      if p.original_colors == 1 then
        disp = original_colors(content_image, disp)
      end

      image.save(filename, disp)
    end
  end

   local function maybe_print(t, loss)
     local verbose = (p.print_iter > 0 and t % p.print_iter == 0)
     if verbose then
       print(string.format('Iteration %d / %d', t, p.num_iterations))
       for i, loss_module in ipairs(p.content_losses) do
         print(string.format('  Content %d loss: %f', i, loss_module.loss))
       end
       for i, loss_module in ipairs(p.style_losses) do
         print(string.format('  Style %d loss: %f', i, loss_module.loss))
       end
       print(string.format('  Total loss: %f', loss))
     end
   end

  local function feval(x)
	num_calls = num_calls +1
	p.net:forward(x)
	local grad = p.net:updateGradInput(x, dy)
	local loss = 0
	for _, mod in ipairs(p.content_losses) do
	  loss = loss + mod.loss
	end
	for _, mod in ipairs(p.style_losses) do
	  loss = loss + mod.loss
	end
	maybe_print(num_calls, loss)
	maybe_save(num_calls)
	collectgarbage()
	return loss ,grad:view(grad:nElement())
  end

  for t = 1, p.num_iterations do
	print(t)
	local x, losses = optim.adam(feval, p.init_image, optim_state)
 end





end
