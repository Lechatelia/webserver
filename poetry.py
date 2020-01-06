#!/usr/bin/env python
# coding: utf-8

import sys
# sys.path.append('../..')
sys.path.append('/data/codes/poetryseq2seqdevelop')

import json

from data_utils import prepare_batch_predict_data
from model import Seq2SeqModel
from vocab import get_vocab, ints_to_sentence
import tensorflow.compat.v1 as tf
from Multilabel import  MultiLabel
from  plan import  Planner
from functools import reduce



# Data loading parameters
tf.app.flags.DEFINE_boolean('rev_data', True, 'Use reversed training data')
tf.app.flags.DEFINE_boolean('align_data', True, 'Use aligned training data')
tf.app.flags.DEFINE_boolean('prev_data', True, 'Use training data with previous sentences')
tf.app.flags.DEFINE_boolean('align_word2vec', True, 'Use aligned word2vec model')

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 1, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', 80, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', 500, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('model_path', None, 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('model_dir', None, 'Path to load model checkpoints')
tf.app.flags.DEFINE_string('predict_mode', 'greedy', 'Decode helper to use for predicting')
tf.app.flags.DEFINE_string('decode_input', '/data/codes/poetryseq2seqdevelop/data/newstest2012.bpe.de', 'Decoding input path')
tf.app.flags.DEFINE_string('decode_output', '/data/codes/poetryseq2seqdevelop/data/newstest2012.bpe.de.trans', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')


FLAGS = tf.app.flags.FLAGS


#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_config(FLAGS):
    if FLAGS.model_path is not None:
        checkpoint_path = FLAGS.model_path
        print ('Model path specified at: {}'.format(checkpoint_path))
    elif FLAGS.model_dir is not None:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir + '/')
        print ('Model dir specified, using the latest checkpoint at: {}'.format(checkpoint_path))
    else:
        checkpoint_path = tf.train.latest_checkpoint('/data/codes/poetryseq2seqdevelop/model/')
        print ('Model path not specified, using the latest checkpoint at: {}'.format(checkpoint_path))

    FLAGS.model_path = checkpoint_path

    # Load config saved with model
    config_unicode = json.load(open('%s.json' % FLAGS.model_path, 'rb'))
    # config = unicode_to_utf8(config_unicode) change by zjg 20191218
    config = config_unicode

    # Overwrite flags
    for key, value in FLAGS.__flags.items():
        config[key] = value.value #change by zjg 20191218
        # config[key] = value

    return config


def load_model(session, model, saver):
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print ('Reloading model parameters..')
        model.restore(session, saver, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


class Seq2SeqPredictor:
    def __init__(self,gpu_number=0):
        # Load model config
        config = load_config(FLAGS)

        config_proto = tf.ConfigProto(
            # allow_soft_placement=FLAGS.allow_soft_placement,
            # log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=True) #, visible_device_list=str(gpu_number))
        )
        self.graphpre = tf.Graph()
        self.sess = tf.Session(graph=self.graphpre, config=config_proto)

        with self.sess.as_default():
            with self.graphpre.as_default():


                # Build the model
                self.model = Seq2SeqModel(config, 'predict')

                # Create saver
                # Using var_list = None returns the list of all saveable variables
                saver = tf.train.Saver(var_list=None)

                # Reload existing checkpoint
                load_model(self.sess, self.model, saver)
                self.planner = Planner()

                print("poetry is ok!")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def Normalize(self, list): #add by zjg
        string = ''
        for i in list:
            string=string+i+' '
        keywords = self.planner.plan(string)
        return keywords



    def predict(self, keywords):
        sentences = []
        keywords = self.Normalize(keywords) # add by zjg
        for keyword in keywords:
            source, source_len = prepare_batch_predict_data(keyword,
                                                            previous=sentences,
                                                            prev=FLAGS.prev_data,
                                                            rev=FLAGS.rev_data,
                                                            align=FLAGS.align_data)
            with self.sess.as_default():
                with self.graphpre.as_default():
                    predicted_batch = self.model.predict(
                        self.sess,
                        encoder_inputs=source,
                        encoder_inputs_length=source_len
                    )

            predicted_line = predicted_batch[0] # predicted is a batch of one line
            predicted_line_clean = predicted_line[:-1] # remove the end token
            predicted_ints = map(lambda x: x[0], predicted_line_clean) # Flatten from [time_step, 1] to [time_step]
            predicted_sentence = ints_to_sentence(predicted_ints)

            if FLAGS.rev_data:
                predicted_sentence = predicted_sentence[::-1]

            sentences.append(predicted_sentence)
        return sentences

def is_quatrain( poem):
        return len(poem) == 4 and \
               (len(poem[0]) == 5 or len(poem[0]) == 7) and \
               reduce(lambda x, line: x and len(line) == len(poem[0]), poem[1:], True)

if __name__ == '__main__':
    predictor = Seq2SeqPredictor()
    multilabel = MultiLabel()
    results = multilabel.test("../pairwiseranking/test.jpg")
    key_words = [i['name'] for i in results]
    KEYWORDS = [
        u'楚',
        u'相思',
        u'收拾',
        u'思乡',
        u'相随'
    ]
    # lines = predictor.predict(KEYWORDS)
    while True:
        lines = predictor.predict(key_words)
        if is_quatrain(lines):
            break
    for line in lines:
        print(line)

