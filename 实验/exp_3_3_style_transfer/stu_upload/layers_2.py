import numpy as np
import struct
import os
import time


class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] * self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size]) + self.bias[idxc]

        self.forward_time = time.time() - start_time
        return self.output

    def forward_speedup(self, input):
        
        self.input = input
        start_time = time.time()
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2

        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding: self.padding + self.input.shape[2], self.padding: self.padding + self.input.shape[3]] = self.input

        h_t = (height - self.kernel_size) / self.stride + 1
        w_t = (width - self.kernel_size) / self.stride + 1


        ###################
        # self.input_col = img2col(self.input_pad, height_out, width_out, self.kernel_size, self.stride)
        # def img2col(input, height_out, width_out, kernel_size, stride):
        self.input_col = np.zeros([self.input_pad.shape[0], self.input_pad.shape[1], self.kernel_size * self.kernel_size, h_t * w_t])
        height = (self.input_pad.shape[2] - self.kernel_size) / self.stride + 1
        width = (self.input_pad.shape[3] - self.kernel_size) / self.stride + 1
        for idxh in range(height):
            for idxw in range(width):
                self.input_col[:, :, :, idxh * width + idxw] = self.input_pad[:, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size].reshape(self.input_pad.shape[0], self.input_pad.shape[1], -1)

        # self.input_col = np.zeros([self.input_pad.shape[0], self.input_pad.shape[1], self.kernel_size * self.kernel_size, ((self.input.shape[2] + self.padding * 2 - self.kernel_size) / self.stride + 1) * ((self.input.shape[3] + self.padding * 2 - self.kernel_size) / self.stride + 1)])
        # height_t = (self.input_pad.shape[2] - self.kernel_size) / self.stride + 1
        # width_t = (self.input_pad.shape[3] - self.kernel_size) / self.stride + 1
        # for h in range(height_t):
        #     for w in range(width_t):
        #         self.input_col[:, :, :, h * width_t + w] = self.input_pad[:, :, h * self.stride: h * self.stride + self.kernel_size, w * self.stride: w * self.stride + self.kernel_size].reshape(self.input_pad.shape[0], self.input_pad.shape[1], -1)

        ##############################

        self.weights_col = self.weight.transpose(3, 0, 1, 2).reshape(self.weight.shape[-1], -1)
        self.output = (np.matmul(self.weights_col, self.input_col.reshape(self.input_col.shape[0], -1, self.input_col.shape[3])) + self.bias.reshape(-1, 1)).reshape(
            self.input.shape[0],
            self.channel_out,
            h_t,
            w_t
        )
        self.forward_time = time.time() - start_time
        return self.output

    def backward_speedup(self, top_diff):
        
        start_time = time.time()
        bottom_diff_col = np.matmul(self.weights_col.T, top_diff.transpose(1, 2, 3, 0).reshape(self.channel_out, -1))
        bottom_diff_col = bottom_diff_col.reshape(bottom_diff_col.shape[0], -1, self.input.shape[0]).transpose(2, 0, 1)

        bottom_diff = np.zeros([bottom_diff_col.shape[0], self.channel_in, self.input.shape[2] + self.padding * 2, self.input.shape[3] + self.padding * 2])
        input = bottom_diff_col.reshape(bottom_diff_col.shape[0], self.channel_in, -1, bottom_diff_col.shape[2])
        height = (self.input.shape[2] + self.padding * 2 - self.kernel_size) / self.stride + 1
        width = (self.input.shape[3] + self.padding * 2 - self.kernel_size) / self.stride + 1
        for h in range(height):
            for w in range(width):
                bottom_diff[:, :, h * self.stride : h * self.stride + self.kernel_size, w * self.stride: w * self.stride + self.kernel_size] += input[:, :, :, h * width + w].reshape(input.shape[0], self.channel_in, self.kernel_size, -1)

        self.backward_time = time.time() - start_time
        return bottom_diff[:, :, self.padding: self.input.shape[2] + self.padding, self.padding: self.input.shape[3] + self.padding]

    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        
                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, idxh * self.stride: idxh * self.stride + self.kernel_size, idxw * self.stride: idxw * self.stride + self.kernel_size]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        bottom_diff = bottom_diff[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        self.backward_time = time.time() - start_time
        return bottom_diff

    def get_gradient(self):
        return self.d_weight, self.d_bias

    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def get_forward_time(self):
        return self.forward_time

    def get_backward_time(self):
        return self.backward_time


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1:  
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))

    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc, idxh * self.stride: idxh * self.stride + self.kernel_size, idxw * self.stride: idxw * self.stride + self.kernel_size])
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh * self.stride: idxh * self.stride + self.kernel_size, idxw * self.stride: idxw * self.stride + self.kernel_size])
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output

    def forward_speedup(self, input):
        
        start_time = time.time()

        self.input = input  # [N, C, H, W]
        self.height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        self.width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1

        output = np.zeros([self.input.shape[0], self.input.shape[1], self.kernel_size * self.kernel_size, self.height_out * self.width_out])
        height = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        for h in range(height):
            for w in range(width):
                output[:, :, :, h * width + w] = input[:, :, h * self.stride: h * self.stride + self.kernel_size, w * self.stride : w * self.stride + self.kernel_size].reshape(input.shape[0], input.shape[1], -1)

        output = output.reshape(self.input.shape[0], self.input.shape[1], -1, self.height_out, self.width_out).max(axis=2, keepdims=True)
        self.max_index = (self.input_col == output)
        self.output = output.reshape(self.input.shape[0], self.input.shape[1], self.height_out, self.width_out)

        return self.output

    def backward_speedup(self, top_diff):
        

        pool_diff = (self.max_index * top_diff[:, :, np.newaxis, :, :]).reshape(self.input.shape[0], -1, self.height_out * self.width_out)

        output = np.zeros([pool_diff.shape[0], self.input.shape[1], self.input.shape[2], self.input.shape[3]])
        pool_diff = pool_diff.reshape(pool_diff.shape[0], self.input.shape[1], -1, pool_diff.shape[2])
        height = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        for h in range(height):
            for w in range(width):
                output[:, :, h * self.stride: h * self.stride + self.kernel_size, w * self.stride: w * self.stride + self.kernel_size] += pool_diff[:, :, :, h * width + w].reshape(pool_diff.shape[0], self.input.shape[1], self.kernel_size, -1)
        bottom_diff = output[:, :, : self.input.shape[2], : self.input.shape[3]]
        return bottom_diff

    def backward_raw_book(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        
                        max_index = np.argwhere(self.max_index[idxn, idxc, idxh * self.stride: idxh * self.stride + self.kernel_size, idxw * self.stride: idxw * self.stride + self.kernel_size])[0]
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw]
        return bottom_diff


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))

    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output

    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
