
import lasagne

import theano.tensor as T
import theano

class ComplexMultLayer(lasagne.layers.MergeLayer):
    """
    Multiplies the inputs together assuming they are complex numbers stored
    in real and imaginary parts along the last dimension

    Parameters
    ----------

    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes
    """

    def __init__(self, a, b, **kwargs):
        super(ComplexMultLayer, self).__init__([a, b], **kwargs)

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(*input_shapes))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return output_shape

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs

        s = T.shape(x)
        x2 = T.reshape(x, (-1,2))
        y2 = T.reshape(y, (-1,2))
        a = x2[:,0]*y2[:,0] - x2[:,1]*y2[:,1]
        b = x2[:,0]*y2[:,1] + x2[:,1]*y2[:,0]

        out = T.reshape(T.stack([a, b], axis=0).T, s)
        return out

class CondConcatLayer(lasagne.layers.MergeLayer):
    """
    Concatenates multiple inputs along the specified axis. Inputs should have
    the same shape except for the dimension specified in axis, which can have
    different sizes.
    Parameters
    -----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes
    axis : int
        Axis which inputs are joined over
    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`. Cropping is always disabled for `axis`.
    """
    def __init__(self, main, branch_0, branch_1, axis=1, **kwargs):
        super(CondConcatLayer, self).__init__([main, branch_0, branch_1], **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        input_shapes = lasagne.layers.merge.autocrop_array_shapes(input_shapes[:-1], None)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
            return (len(shape1) == len(shape2) and
                    all(i == axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def get_output_for(self, inputs, alternative_path=False, **kwargs):
        main, branch_0, branch_1 = inputs
        if alternative_path:
            return T.concatenate([main, branch_1], axis=self.axis)
        else:
            return T.concatenate([main, branch_0], axis=self.axis)
