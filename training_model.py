import tensorflow as tf

"""gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)]
    )

print(tf.config.experimental.get_memory_info("GPU:0"))"""

import sys
import gc  
from custom_transformers_model import CustomSchedule, Transformer, Translator, masked_loss, masked_accuracy
from tokenizers import Tokenizer
from tokenizening_data import tokenizing_text, load_dataset, parse_function
data = sys.argv[1]


batch_size = 8
tfrecord_file = "dataset.tfrecord"


#maxlen = tokenizing_text(data, batch_size)

dataset = load_dataset("dataset.tfrecord", batch_size)
print("Dataset loaded successfully.\n")

tokenizer = Tokenizer.from_file("tokenizer")

for (encoder_input, decoder_input), decoder_output in dataset.take(1000):
    print("Encoder Input:")
    print(encoder_input.numpy())
    print("\nDecoder Input:")
    print(decoder_input.numpy())
    print("\nDecoder Output:")
    print(decoder_output.numpy())



num_layers = 6
d_model = 128
dff = 128
num_heads = 6
dropout_rate = 0.1

transformers = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    dff=dff,
    num_heads=num_heads,
    input_vocab_size=tokenizer.get_vocab_size(),
    target_vocab_size=tokenizer.get_vocab_size(),
    dropout_rate=dropout_rate
)


learning_rate = CustomSchedule(d_model=128)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformers.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])


"""dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(parse_function).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#dataset = dataset.take(100)"""


gc.collect()
tf.keras.backend.clear_session()

print(f"\nStarting Training with Batch Size = {batch_size}\n")
transformers.fit(dataset, epochs=25)  


transformers.save("lang_transformers.keras")
print(" The Transformers model has been saved.")
