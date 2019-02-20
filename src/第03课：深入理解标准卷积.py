'''
@Desc  : 
@Date  : 2019/2/20
@Author: zhangjianfeng 
'''

net_struct = {
    'alexnet': {'net': [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0]],
                'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']},
    'vgg16': {'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0]],
              'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                       'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3',
                       'pool5']},
    'zf-5': {'net': [[7, 2, 3], [3, 2, 1], [5, 2, 2], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
             'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5']}}
imsize = 224


def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2 * pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride


def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF - 1) * stride) + fsize
    return RF


if __name__ == '__main__':
    print("layer output sizes given image = %dx%d" % (imsize, imsize))
for net in net_struct.keys():
    print('************net structrue name is %s**************' % net)
    for i in range(len(net_struct[net]['net'])):
        p = outFromIn(imsize, net_struct[net]['net'], i + 1)
        rf = inFromOut(net_struct[net]['net'], i + 1)
        print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (
            net_struct[net]['name'][i], p[0], p[1], rf))

print("===============================================================================")
import tensorflow as tf

img1 = tf.constant(value=[[[[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]], [[1], [2], [3], [4]]]],
                   dtype=tf.float32)
img2 = tf.constant(value=[[[[1], [1], [1], [1]], [[1], [1], [1], [1]], [[1], [1], [1], [1]], [[1], [1], [1], [1]]]],
                   dtype=tf.float32)
img = tf.concat(values=[img1, img2], axis=3)
filter1 = tf.constant(value=0, shape=[3, 3, 1, 1], dtype=tf.float32)
filter2 = tf.constant(value=1, shape=[3, 3, 1, 1], dtype=tf.float32)
filter3 = tf.constant(value=2, shape=[3, 3, 1, 1], dtype=tf.float32)
filter4 = tf.constant(value=3, shape=[3, 3, 1, 1], dtype=tf.float32)
filter_out1 = tf.concat(values=[filter1, filter2], axis=2)
filter_out2 = tf.concat(values=[filter3, filter4], axis=2)
filter = tf.concat(values=[filter_out1, filter_out2], axis=3)

point_filter = tf.constant(value=1, shape=[1, 1, 4, 4], dtype=tf.float32)
out_img = tf.nn.depthwise_conv2d(input=img, filter=filter, strides=[1, 1, 1, 1], rate=[1, 1], padding='VALID')
out_img = tf.nn.conv2d(input=out_img, filter=point_filter, strides=[1, 1, 1, 1], padding='VALID')
