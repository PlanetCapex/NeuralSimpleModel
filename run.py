import mxnet as mx
import model
import numpy as np
from collections import namedtuple
from skimage import io, transform


content_image = 'input/content_img.jpg'
style_image = 'input/style_img.jpg'
stop_stage = 0.0005
max_num_stage = 1000
output = 'output/out.jpg'



def PreContentImage(path, long_edge):
    img = io.imread(path)
    factor = float(long_edge) / max(img.shape[:2])
    new_size = (int(img.shape[0] * factor), int(img.shape[1] * factor))
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PreStyleImage(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PostImage(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename):
    print('Готово ', filename)
    out = PostImage(img)
    io.imsave(filename, out)


dev = mx.gpu()
content_np = PreContentImage(content_image, 1920)
style_np = PreStyleImage(style_image, shape=content_np.shape)
size = content_np.shape[2:]


Executor = namedtuple('Executor', ['executor', 'data', 'data_grad'])


def get_loss(gram, content):
    gram_loss = []
    for i in range(len(gram.list_outputs())):
        gvar = mx.sym.Variable("target_gram_%d" % i)
        gram_loss.append(mx.sym.sum(mx.sym.square(gvar - gram[i])))
    cvar = mx.sym.Variable("target_content")
    content_loss = mx.sym.sum(mx.sym.square(cvar - content))
    return mx.sym.Group(gram_loss), content_loss


def style_gram_symbol(input_size, style):
    _, output_shapes, _ = style.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    gram_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]
        x = mx.sym.Reshape(style[i], target_shape=(int(shape[1]), int(np.prod(shape[2:]))))
        
        gram = mx.sym.FullyConnected(x, x, no_bias=True, num_hidden=shape[1])
        gram_list.append(gram)
        grad_scale.append(np.prod(shape[1:]) * shape[1])
    return mx.sym.Group(gram_list), grad_scale







model_module =  model
style, content = model_module.get_symbol()
gram, gscale = style_gram_symbol(size, style)
model_executor = model_module.get_executor(gram, content, size, dev)
model_executor.data[:] = style_np
model_executor.executor.forward()
style_array = []
for i in range(len(model_executor.style)):
    style_array.append(model_executor.style[i].copyto(mx.cpu()))

model_executor.data[:] = content_np
model_executor.executor.forward()
content_array = model_executor.content.copyto(mx.cpu())
del model_executor


style_loss, content_loss = get_loss(gram, content)
model_executor = model_module.get_executor(
    style_loss, content_loss, size, dev)

grad_array = []
for i in range(len(style_array)):
    style_array[i].copyto(model_executor.arg_dict["target_gram_%d" % i])
    grad_array.append(mx.nd.ones((1,), dev) * (float(1) / gscale[i]))
grad_array.append(mx.nd.ones((1,), dev) * (float(10)))

print([x.asscalar() for x in grad_array])
content_array.copyto(model_executor.arg_dict["target_content"])


img = mx.nd.zeros(content_np.shape, ctx=dev)
img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)

lr = mx.lr_scheduler.FactorScheduler(step=80, factor=.9)

optimizer = mx.optimizer.SGD(
    learning_rate = 0.001,
    wd = 0.0005,
    momentum=0.1,
    lr_scheduler = lr)
optim_state = optimizer.create_state(0, img)

old_img = img.copyto(dev)
c_norm = 1 * np.prod(img.shape)


for e in range(max_num_stage):
    img.copyto(model_executor.data)
    model_executor.executor.forward()
    model_executor.executor.backward(grad_array)
    gnorm = mx.nd.norm(model_executor.data_grad).asscalar()
    if gnorm > clip_norm:
        model_executor.data_grad[:] *= c_norm / gnorm


    optimizer.update(0, img, model_executor.data_grad, optim_state)
    new_img = img
    stage = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img)).asscalar()

    old_img = new_img.copyto(dev)
    print('этап ' , e , ' коэффициент изменения', stage)
    if stage < stop_stage:
        print('Завершено.')
        break

'''End'''
SaveImage(new_img.asnumpy(), output)




