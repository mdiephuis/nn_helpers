import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO
from PIL import Image


class TFLogger(object):
    def __init__(self, fp):
        self.writer = tf.summary.FileWriter(fp)
        # self.writer = tf.contrib.summary.create_file_writer(fp)

    def add_scalar(self, tag, value, step):
        '''
        Log a scalar value
        '''
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def add_image(self, tag, image, step):
        '''
        Log an image
        '''
        summary = tf.contrib.summary.image(tag, image, step)
        self.writer.add_summary(summary, step)

    def add_histogram(self, tag, values, step, bins=1000):
        '''
        Log a histogram
        '''
        counts, bin_edges = np.histogram(values, bins=bins)

        # Build tf histogram proto
        tf_hist = tf.HistogramProto()
        tf_hist.min = float(np.min(values))
        tf_hist.max = float(np.max(values))
        tf_hist.num = int(np.prod(values.shape))
        tf_hist.sum = float(np.sum(values))
        tf_hist.sum_squares = float(np.sum(values ** 2))

        # drop start of first bin
        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            tf_hist.bucket_limit.append(edge)
        for count in counts:
            tf_hist.bucket.append(count)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=tf_hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
