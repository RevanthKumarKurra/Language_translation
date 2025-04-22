
import tensorflow as tf
import sys
import numpy as np
from tokenizers import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = sys.argv[1]
tokenizer = Tokenizer.from_file("tokenizer")

def tokenizing_text(data, batch_size, tfrecord_file="dataset.tfrecord"):
    print("Reading and loading text")
    with open(f"./{data}", "r", encoding="utf-8") as txt:
        text = txt.readlines()

    text_list = [line.split("\t\t") for line in text]

    print("Calculating max sequence length")
    maxlen = max(
        max(len(tokenizer.encode(item[0]).ids) for item in text_list),
        max(len(tokenizer.encode(item[1].strip()).ids) for item in text_list)
    )
    print(f"Max sequence length: {maxlen}")

    def serialize_example(encoder_input, decoder_input, decoder_output):
       
        encoder_input = [float(x) for x in encoder_input]
        decoder_input = [float(x) for x in decoder_input]
        decoder_output = [float(x) for x in decoder_output]
        feature = {
            'encoder_input': tf.train.Feature(float_list=tf.train.FloatList(value=encoder_input)),
            'decoder_input': tf.train.Feature(float_list=tf.train.FloatList(value=decoder_input)),
            'decoder_output': tf.train.Feature(float_list=tf.train.FloatList(value=decoder_output))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    

    print("Writing TFRecord file")
    with tf.io.TFRecordWriter(tfrecord_file) as writer:

        for item in text_list:
           
            encoder_input = tokenizer.encode(item[0]).ids
            output_seq = tokenizer.encode(item[1].strip()).ids

            
            encoder_input = pad_sequences([encoder_input], maxlen=maxlen, padding="post", dtype="int32")[0]
            output_seq = pad_sequences([output_seq], maxlen=maxlen, padding="post", dtype="int32")[0]

            
            decoder_input = np.where(np.isin(output_seq, [7, 8, 9, 10]), 0, output_seq)

            
            decoder_output = np.zeros_like(output_seq, dtype=np.int32)
            decoder_output[:-1] = output_seq[1:]
            decoder_output[-1] = 0

            writer.write(serialize_example(encoder_input, decoder_input, decoder_output))

    
    print(f"TFRecord file saved: {tfrecord_file}")
    return maxlen


def parse_function(proto):
    feature_description = {
        'encoder_input': tf.io.VarLenFeature(tf.float32),
        'decoder_input': tf.io.VarLenFeature(tf.float32),
        'decoder_output': tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(proto, feature_description)
    encoder_input = tf.sparse.to_dense(example['encoder_input'])
    decoder_input = tf.sparse.to_dense(example['decoder_input'])
    decoder_output = tf.sparse.to_dense(example['decoder_output'])

    return (encoder_input, decoder_input), decoder_output


def load_dataset(tfrecord_file, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
