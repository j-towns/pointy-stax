import nn_jax as nn


_zero_pad = (0, 0, 0)

# def model_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu', energy_distance=False):
def PixelCNNPP(dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10,
               resnet_nonlinearity=nn.ConcatElu):
    def pixel_cnn(x):
        """
        From the original docstring:
        We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
        a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
        of the x_out tensor describes the predictive distribution for the RGB at
        that position.
        """
#       # ////////// up pass through pixelCNN ////////
#       x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on
        x_pad = nn.Pad([_zero_pad, _zero_pad, _zero_pad, (0, 1, 0)], 1.)(x)

#       u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
        u_list = [nn.DownShift(nn.DownShiftedConv2D(out_chan=nr_filters, filter_shape=[2, 3])(x_pad))]

#       ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
        ul_list = [nn.FanInSum((nn.DownShift(nn.DownShiftedConv2D(out_chan=nr_filters, filter_shape=[1, 3])(x_pad)), # None
#                  nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                                nn.RightShift(nn.DownRightShiftedConv2D(out_chan=nr_filters, filter_shape=[2, 1])(x_pad))))]

#       for rep in range(nr_resnet):
        for _ in range(nr_resnet):
#           u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            u_list.append(nn.GatedResnet(conv=nn.DownShiftedConv2D, dropout_p=dropout_p)(u_list[-1]))
#           ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
            ul_list.append(nn.FanInGatedResnet(conv=nn.DownRightShiftedConv2D, dropout_p=dropout_p)((ul_list[-1], u_list[-1])))

#       u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=[2, 2]))
        u_list.append(nn.DownShiftedConv2D(out_chan=nr_filters, strides=[2, 2])(u_list[-1]))
#       ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=[2, 2]))
        ul_list.append(nn.DownRightShiftedConv2D(out_chan=nr_filters, strides=[2, 2])(ul_list[-1]))

#       for rep in range(nr_resnet):
        for _ in range(nr_resnet):
#           u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            u_list.append(nn.GatedResnet(conv=nn.DownShiftedConv2D, dropout_p=dropout_p)(u_list[-1]))
#           ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
            ul_list.append(nn.FanInGatedResnet(conv=nn.DownRightShiftedConv2D, dropout_p=dropout_p)((ul_list[-1], u_list[-1])))

#       u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, strides=[2, 2]))
        u_list.append(nn.DownShiftedConv2D(out_chan=nr_filters, strides=[2, 2])(u_list[-1]))
#       ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, strides=[2, 2]))
        ul_list.append(nn.DownRightShiftedConv2D(out_chan=nr_filters, strides=[2, 2])(ul_list[-1]))

#       for rep in range(nr_resnet):
        for _ in range(nr_resnet):
#           u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            u_list.append(nn.GatedResnet(conv=nn.DownShiftedConv2D, dropout_p=dropout_p)(u_list[-1]))
#           ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
            ul_list.append(nn.FanInGatedResnet(conv=nn.DownRightShiftedConv2D, dropout_p=dropout_p)((ul_list[-1], u_list[-1])))

#       # /////// down pass ////////
#       u = u_list.pop()
        u = u_list.pop()
#       ul = ul_list.pop()
        ul = ul_list.pop()

#       for rep in range(nr_resnet):
        for _ in range(nr_resnet):
#           u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            u = nn.FanInGatedResnet(conv=nn.DownShiftedConv2D, dropout_p=dropout_p)((u, u_list.pop()))
#           ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
            ul = nn.FanInGatedResnet(conv=nn.DownRightShiftedConv2D, dropout_p=dropout_p)((ul, nn.FanInConcat()((u, ul_list.pop()))))

#       u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, strides=[2, 2])
        u = nn.DownShiftedDeConv2D(out_chan=nr_filters, strides=[2, 2])(u)
#       ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=[2, 2])
        ul = nn.DownRightShiftedDeConv2D(out_chan=nr_filters, strides=[2, 2])(ul)
#
#       for rep in range(nr_resnet+1):
        for _ in range(nr_resnet + 1):
#           u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            u = nn.FanInGatedResnet(conv=nn.DownShiftedConv2D, dropout_p=dropout_p)((u, u_list.pop()))
#           ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
            ul = nn.FanInGatedResnet(conv=nn.DownRightShiftedConv2D, dropout_p=dropout_p)((ul, nn.FanInConcat()((u, ul_list.pop()))))

#       u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, strides=[2, 2])
        u = nn.DownShiftedDeConv2D(out_chan=nr_filters, strides=[2, 2])(u)
#       ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, strides=[2, 2])
        ul = nn.DownRightShiftedDeConv2D(out_chan=nr_filters, strides=[2, 2])(ul)
#
#       for rep in range(nr_resnet+1):
        for _ in range(nr_resnet + 1):
#           u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            u = nn.FanInGatedResnet(conv=nn.DownShiftedConv2D, dropout_p=dropout_p)((u, u_list.pop()))
#           ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
            ul = nn.FanInGatedResnet(conv=nn.DownRightShiftedConv2D, dropout_p=dropout_p)((ul, nn.FanInConcat()((u, ul_list.pop()))))

#       x_out = nn.nin(tf.nn.elu(ul), 10*nr_logistic_mix)
        x_out = nn.NIN(10 * nr_logistic_mix)(nn.Elu(ul))
#
#       assert len(u_list) == 0
        assert len(u_list) == 0
#       assert len(ul_list) == 0
        assert len(ul_list) == 0

#       return x_out
        return x_out
    return nn.pointy_to_stax_layer(pixel_cnn)
