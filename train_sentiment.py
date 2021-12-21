import argparse
import logging
import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import json
import horovod.tensorflow.keras as hvd
from sagemaker_tensorflow import PipeModeDataset
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
    
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class SentimentClassifierModel(tf.keras.models.Model):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        self.embedding = tf.keras.layers.Embedding(self.params['max_word_index']+1, 128, input_length=self.params['max_sentence_length'])
        self.conv1d_1 = tf.keras.layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)
        self.maxpool1d_1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten_1 = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(20, activation='relu')
        self.sentiment = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        print("Input Shape = {}".format(inputs.shape))
        x = self.embedding(inputs)
        x = self.conv1d_1(x)
        x = self.maxpool1d_1(x)
        x = self.flatten_1(x)
        x = self.dense_1(x)
        x = self.sentiment(x)
        return x
    
    # Hacky way to specify input shape on subclasses keras model
    def model(self):
        x = tf.keras.layers.Input(shape=(self.params['max_sentence_length']))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
    
    

def parse(record, max_sentence_length):
    # parse and clean tensors in PipeModeDataset
    features = {
        "sentences": tf.io.VarLenFeature(tf.int64),
        "sentiment": tf.io.FixedLenFeature([], tf.int64)
    }
    
    parsed = tf.io.parse_single_example(record, features)
    logger.info("Padded Record: " + str(parsed))
    return parsed

def read_data_pipe():
    dataset = PipeModeDataset(channel='train', record_format='TFRecord')
    dataset = dataset.repeat()
    dataset = dataset.prefetch(10)
    dataset = dataset.map(lambda record: parse(record, max_sentence_length), num_parallel_calls=10)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    return dataset


if __name__ == '__main__':
    logger.info("Starting training")
    parser = argparse.ArgumentParser()
    
    # These are read from parameters the 'hyperparameters' argument of th Estimator
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data-config', type=json.loads, default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    
    # SageMaker automatically sets these environment variables in the container
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR')) 
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args, _ = parser.parse_known_args()
    
    hvd.init()
    print("Number of GPUs: ", hvd.size())
    
    # Pin each GPU to a single process local rank
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        # allocates only as much GPU memory needed for runtime allocations
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        # restrict this process to only run on a specific GPU
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
    callbacks = []
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    
    # Configure and wrap the model optimizer
    opt = optimizers.Adam(learning_rate=0.01)
    opt = hvd.DistributedOptimizer(opt)
        
#     if args.data_config['train']['TrainingInputMode'] == 'Pipe':
#         read_data_pipe()

    from ast import literal_eval

    train = pd.read_csv(args.train + '/train.csv', sep=',', converters=
                        {'sentences': literal_eval})
    
    
    X_train, y_train = train['sentences'], train['sentiment']
        
    max_word_index = max([max(sentence) for sentence in X_train])
    max_sentence_length = max([len(sentence) for sentence in X_train])
    
    X_train_padded = pad_sequences(X_train, maxlen=int(float(max_sentence_length)), padding='post')

    params = dict(
        max_word_index=max_word_index,
        max_sentence_length=max_sentence_length,
        batch_size=args.batch_size
    )
    
    model = SentimentClassifierModel(params).model()
    
    print("Model Summary: ", model.summary())

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train_padded, y_train, batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks)
    
    if hvd.rank() == 0:
        model.save(os.path.join(args.sm_model_dir, '1'), 'sentiment_model.h5')
    
    