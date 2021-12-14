# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import tensorflow as tf
import numpy as np

global_s = tf.Variable(0, name='global_s', trainable=False)
global_update = tf.Variable(0, name='global_s', trainable=False)


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


def generateRandomIndices(shape):
    """generate random indices for a tensor of size (30% of the received shape)"""
    dim = shape
    tensor_size = tf.math.reduce_prod(dim).numpy()
    stf = tf.cast(tensor_size * 0.3, dtype=tf.int32) + 1
    rand = tf.cast(tf.reshape((), (stf, 0)), dtype=tf.int32)  # an empty tensor to start with
    update = int(global_update)  ####<<(check the note below)
    seed = dim[0] * 150 + update
    # constructing a vector of indices for each dimension, and concat them to create the indices vector.
    for i in range(0, len(dim)):
        rng = tf.random.experimental.Generator.from_seed(seed)
        r = rng.uniform((stf, 1), 0, dim[i], dtype=tf.int32)
        tf.transpose(r)
        rand = tf.concat([rand, r], -1)
        seed += dim[i]
    ####################################################################################################################
    # (NOTE#1):
    # The seed need to be updated after every epoch, because we want pick another set of random indices for every epoch.
    # (Any update can work, for example incrementing the seed by one, or whatever)
    # TF2 no longer support using global-steps
    # For now, I am using the class global variables and assuming that we have 2 workers.
    # This need to be tested in Horovod,
    # and if those variables did not work, we can try using the iteration# callback ,for example, as an update,
    # which should be common among all workers.
    ####################################################################################################################
    return rand


class RandomCompressor(Compressor):
    """Compress the gradients by randomly selecting 30% of the tensor."""

    @staticmethod
    def compress(tensor):
        """returns 30% of the tensor."""
        shape = tensor.get_shape()
        ind = generateRandomIndices(shape)
        result = tf.gather_nd(tensor, ind)
        global_s.assign_add(1)
        return result, shape

    @staticmethod
    def decompress(tensor, ctx):
        """reconstruct the tensor to the original shape and fill the empty places with zeros."""
        dim = ctx  # The shape of the tensor to be reconstructed
        stf = tensor.get_shape()[0]  # the number of elements in each dimension
        ind = generateRandomIndices(dim)
        orgA = np.zeros(dim)
        for j in range(stf):
            orgA[ind[j]] = tensor[j]
        org = tf.convert_to_tensor(orgA)

        if global_s % 2 == 0:  # to take care of NOTE#1
            global_update.assign_add(1)

        return org


class NoneCompressor(Compressor):
    """Default no-op compression."""

    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating:
            tensor_decompressed = tf.cast(tensor, dtype=dtype)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor

    """Compress the gradients by randomly selecting 30% of the tensor."""
    rand = RandomCompressor
