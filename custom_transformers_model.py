import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# This is the function to calculate the position of the words in a sentence and convert them into vector
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]    
  depths = np.arange(depth)[np.newaxis, :]/depth   

  angle_rates = 1 / (10000**depths)         
  angle_rads = positions * angle_rates      

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
  


# This is the BaseAttention from this we can Define the MultiHead Attention,CrossAttention etc

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


# CrossAttention 
"""This is the Attention that Which will Encoder context vectors and the till Decoder predicte sentence to predict the word or to check how much 
the word is similar to decoder weight so it gone take key and value vector from the Encoder"""
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  

# GlobalSelf Attention
"""This is the Attention block that which we use in the Encoder Block"""

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  


# CausalSelf Attention
"""The CausalSelf Attention block that which we use in the Decoder Block"""

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


# FeedForward Neural Network

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
  
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
  

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,*,d_model,num_heads,dff,dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
  

  
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):

    x = self.pos_embedding(x) 

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    return x
  


import keras
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x  = inputs
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
    
    def get_config(self):
        """Required for model saving & loading."""
        return {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "dropout_rate": self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
  def get_config(self):  # Required for serialization
    
    return {"d_model": self.d_model.numpy(), "warmup_steps": self.warmup_steps}

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  

learning_rate = CustomSchedule(d_model=128)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)



@tf.keras.utils.register_keras_serializable()

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


@tf.keras.utils.register_keras_serializable()

def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

"""transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])"""

class Translator:
    def __init__(self, tokenizer, transformer,start,end):
        self.tokenizer = tokenizer

        self.transformer = transformer
        self.start = start
        self.end = end
        

    def __call__(self, sentence, maxlen):
        
        # Tokenize and pad the sentence
        self.maxlen = maxlen
        sentence = self.tokenizer.encode(sentence).ids
        print(sentence)
        print(self.start, self.end)
        print(self.tokenizer.encode(self.start).ids[0] and self.tokenizer.encode(self.end).ids[0])
        sentence = pad_sequences([sentence], maxlen=self.maxlen, padding='post')
        sentence = tf.convert_to_tensor(sentence)  # Convert to tensor with correct shape

        # Ensure sentence shape is (batch_size, sequence_length)
        if len(sentence.shape) == 1:
            sentence = tf.expand_dims(sentence, 0)

        encoder_input = sentence  # Shape should now be (1, maxlength)

        #start = self.marathi_tokenizer.word_index['start']
        #end = self.marathi_tokenizer.word_index['end']

        start = self.tokenizer.encode(self.start).ids[0] 
        end = self.tokenizer.encode(self.end).ids[0] 

        print(start,end)

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
        
        for i in tf.range(self.maxlen):
            output = tf.transpose(output_array.stack())

            # Ensure output shape is (batch_size, sequence_length)
            if len(output.shape) == 1:
                output = tf.expand_dims(output, 0)

            predictions = self.transformer([encoder_input, output], training=False)

         # Select the last token from the seq_len dimension.
            predictions = predictions[:, -1:, :]  # Shape (batch_size, 1, vocab_size).

            predicted_id = tf.argmax(predictions, axis=-1)

        # Extract the scalar from predicted_id[0] to match the expected shape.
            output_array = output_array.write(i + 1, predicted_id[0, 0])

            if tf.equal(predicted_id[0, 0], end):
                break

        output = tf.transpose(output_array.stack())
        if len(output.shape) == 1:
            output = tf.expand_dims(output,0)

        
        tokens = []
        text = ""
        for token in output.numpy()[0]:

            if token != start and end:
                word = self.tokenizer.decode([token])
                tokens.append(word)
                text += word+" "
                text = re.sub(r"(?<=\S)\s(?=\S)", "",text)

        # Get attention weights from the transformer after re-running it
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text.strip(), tokens, attention_weights