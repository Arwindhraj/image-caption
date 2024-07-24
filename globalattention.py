"""
Original file is located at
    https://colab.research.google.com/drive/1WFLVw-F_WKPIftolw39Ka9erb0Tblh1R

Dataset 
    Downloading the data
    https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
    https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import load_img
import string
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import Input
from PIL import Image
from tqdm import tqdm

import glob
from gtts import gTTS
from playsound import playsound
from IPython import display
import collections

import numpy as np
import pandas as pd
import os
import imageio
from PIL import Image
import random

text_file = 'D:/Project_Files/image-caption/Flicker8k Dataset/captions.txt'
img_path = 'D:/Project_Files/image-caption/Flicker8k Dataset/Flicker8k_Dataset/'

all_img_id = []
all_img_vector = []
annotations = []

with open(text_file , 'r') as fo:
  next(fo)
  for line in fo :
    split_arr = line.split(',')
    all_img_id.append(split_arr[0])
    annotations.append(split_arr[1].rstrip('\n.'))
    all_img_vector.append(img_path+split_arr[0])

vocabulary = [word.lower() for line in annotations for word in line.split()]
val_count = Counter(vocabulary)
val_count

rem_punct = str.maketrans('', '', string.punctuation)
for r in range(len(annotations)) :
  line = annotations[r]
  line = line.split()
  line = [word.lower() for word in line]
  line = [word.translate(rem_punct) for word in line]
  line = [word for word in line if len(word) > 1]
  line = [word for word in line if word.isalpha()]
  annotations[r] = ' '.join(line)

annotations = ['<start>' + ' ' + line + ' ' + '<end>' for line in annotations]
all_img_path = all_img_vector
top_word_cnt = 5000
tokenizer = Tokenizer(num_words = top_word_cnt+1, filters= '!"#$%^&*()_+.,:;-?/~`{}[]|\=@ ',
                      lower = True, char_level = False,
                      oov_token = 'UNK')
tokenizer.fit_on_texts(annotations)
train_seqs = tokenizer.texts_to_sequences(annotations)
tokenizer.word_index['PAD'] = 0
tokenizer.index_word[0] = 'PAD'
tokenizer.index_word

tokenizer_top_words = [word for line in annotations for word in line.split() ]
tokenizer_top_words_count = collections.Counter(tokenizer_top_words)
tokens = tokenizer_top_words_count.most_common(30)
most_com_words_df = pd.DataFrame(tokens, columns = ['Word', 'Count'])
train_seqs_len = [len(seq) for seq in train_seqs]
longest_word_length = max(train_seqs_len)
cap_vector= tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding= 'post', maxlen = longest_word_length,
                                                          dtype='int32', value=0)
train_seqs_len = [len(seq) for seq in train_seqs]
longest_word_length = max(train_seqs_len)
cap_vector= tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding= 'post', maxlen = longest_word_length,
                                                          dtype='int32', value=0)
preprocessed_image = []
IMAGE_SHAPE = (224, 224)
Display_Images = preprocessed_image[0:5]
figure, axes = plt.subplots(1,5)
figure.set_figwidth(25)
for ax, image in zip(axes, Display_Images) :
  print('Shape after resize : ', image.shape)
  ax.imshow(image)
  ax.grid('off')

def load_images(image_path) :
  img = tf.io.read_file(image_path, name = None)
  img = tf.image.decode_jpeg(img, channels=0)
  img = tf.image.resize(img, IMAGE_SHAPE)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

all_img_vector

training_list = sorted(set(all_img_vector))
New_Img = tf.data.Dataset.from_tensor_slices(training_list)
New_Img = New_Img.map(load_images, num_parallel_calls = tf.data.experimental.AUTOTUNE)
New_Img = New_Img.batch(64, drop_remainder=False)

path_train, path_test, caption_train, caption_test = train_test_split(all_img_vector, cap_vector, test_size = 0.2, random_state = 42)

print("Training data for images: " + str(len(path_train)))
print("Testing data for images: " + str(len(path_test)))
print("Training data for Captions: " + str(len(caption_train)))
print("Testing data for Captions: " + str(len(caption_test)))

image_model = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
image_features_extract_model.summary()
img_features = {}
for image, image_path in tqdm(New_Img) :
  batch_features = image_features_extract_model(image)
  batch_features_flattened = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
  for batch_feat, path in zip(batch_features_flattened, image_path) :
    feature_path = path.numpy().decode('utf-8')
    img_features[feature_path] = batch_feat.numpy()

def map(image_name, caption):
    img_tensor = img_features[image_name.decode('utf-8')]
    return img_tensor, caption

BUFFER_SIZE = 1000
BATCH_SIZE = 32
def gen_dataset(img, capt):

    data = tf.data.Dataset.from_tensor_slices((img, capt))
    data = data.map(lambda ele1, ele2 : tf.numpy_function(map, [ele1, ele2], [tf.float32, tf.int32]),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
    data = (data.shuffle(BUFFER_SIZE, reshuffle_each_iteration= True).batch(BATCH_SIZE, drop_remainder = False)
    .prefetch(tf.data.experimental.AUTOTUNE))
    return data

train_dataset = gen_dataset(path_train,caption_train)
test_dataset = gen_dataset(path_test,caption_test)
sample_img_batch, sample_cap_batch = next(iter(train_dataset))
embedding_dim = 256
units = 512
vocab_size = 5001
train_num_steps = len(path_train) // BATCH_SIZE
test_num_steps = len(path_test) // BATCH_SIZE

max_length = 31
feature_shape = batch_feat.shape[1]
attention_feature_shape = batch_feat.shape[0]

tf.compat.v1.reset_default_graph()
print(tf.compat.v1.get_default_graph())

class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim) 

    def call(self, features):
        features =  self.dense(features)
        features =  tf.keras.activations.relu(features, alpha=0.01, max_value=None, threshold=0)
        return features

encoder=Encoder(embedding_dim)

class GlobalAttention(Model):
    def __init__(self, units):
        super(GlobalAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.units=units

    def call(self, features, hidden):
        hidden_with_time_axis = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=-1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=-1)
        return context_vector, attention_weights

class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attentionglobal = GlobalAttention(self.units) 
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) 
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) 
        self.d2 = tf.keras.layers.Dense(vocab_size) 


    def call(self,x,features, hidden):
        global_context_vector, global_attention_weights = self.attentionglobal(features, hidden)
        embed = self.embed(x) 
        embed = tf.concat([tf.expand_dims(global_context_vector, 1), embed], axis = -1)
        output,state = self.gru(embed) 
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) 
        output = self.d2(output) 
        return output, state, global_attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

decoder=Decoder(embedding_dim, units, vocab_size)

features=encoder(sample_img_batch)

hidden = decoder.init_state(batch_size=sample_cap_batch.shape[0])
dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * sample_cap_batch.shape[0], 1)

predictions, hidden_out, attention_weights= decoder(dec_input, features, hidden)
print('Feature shape from Encoder: {}'.format(features.shape)) 
print('Predcitions shape from Decoder: {}'.format(predictions.shape)) 
print('Attention weights shape from Decoder: {}'.format(attention_weights.shape)) 

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.003)  

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
image_features_extract_model.compile(loss=loss_object, optimizer=optimizer,metrics=['accuracy'])

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    

    return tf.reduce_mean(loss_)

checkpoint_path = "Flickr8K/checkpoint1"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

@tf.function
def train_step(img_tensor, target):
    loss = 0
    hidden = decoder.init_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:

        encoder_op = encoder(img_tensor)
        for r in range(1, target.shape[1]) :
          predictions, hidden, _ = decoder(dec_input, encoder_op, hidden)
          loss = loss + loss_function(target[:, r], predictions)
          dec_input = tf.expand_dims(target[:, r], 1)

    avg_loss = (loss/ int(target.shape[1])) #avg loss per batch
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    grad = tape.gradient (loss, trainable_vars)
    optimizer.apply_gradients(zip(grad, trainable_vars))

    return loss, avg_loss

@tf.function
def test_step(img_tensor, target):
    loss = 0
    hidden = decoder.init_state(batch_size = target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
      encoder_op = encoder(img_tensor)
      for r in range(1, target.shape[1]) :
        predictions, hidden, _ = decoder(dec_input, encoder_op, hidden)
        loss = loss + loss_function(target[:, r], predictions)
        dec_input = tf.expand_dims(target[: , r], 1)
    avg_loss = (loss/ int(target.shape[1])) #avg loss per batch
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    grad = tape.gradient (loss, trainable_vars)
    optimizer.apply_gradients(zip(grad, trainable_vars))
    return loss, avg_loss

def test_loss_cal(test_dataset):
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(test_dataset) :
      batch_loss, t_loss = test_step(img_tensor, target)
      total_loss = total_loss + t_loss
      avg_test_loss = total_loss/ test_num_steps

    return avg_test_loss

loss_plot = []
test_loss_plot = []
EPOCHS = 30
best_test_loss=100
for epoch in tqdm(range(0, EPOCHS)):
    start = time.time()
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(train_dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        avg_train_loss=total_loss / train_num_steps
    loss_plot.append(avg_train_loss)
    test_loss = test_loss_cal(test_dataset)
    test_loss_plot.append(test_loss)
    print ('For epoch: {}, the train loss is {:.3f}, & test loss is {:.3f}'.format(epoch+1,avg_train_loss,test_loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    if test_loss < best_test_loss:
        print('Test loss has been reduced from %.3f to %.3f' % (best_test_loss, test_loss))
        best_test_loss = test_loss
        ckpt_manager.save()

# from matplotlib.pyplot import figure
# figure(figsize=(12, 8))
# plt.plot(loss_plot, color='orange', label = 'training_loss_plot')
# plt.plot(test_loss_plot, color='green', label = 'test_loss_plot')
# plt.xlabel('Epochs', fontsize = 15, color = 'red')
# plt.ylabel('Loss', fontsize = 15, color = 'red')
# plt.title('Loss Plot', fontsize = 20, color = 'red')
# plt.legend()
# plt.show()

# def evaluate(image):
#     attention_plot = np.zeros((max_length, attention_feature_shape))

#     hidden = decoder.init_state(batch_size=1)

#     temp_input = tf.expand_dims(load_images(image)[0], 0)
#     img_tensor_val = image_features_extract_model(temp_input)
#     img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

#     features = encoder (img_tensor_val)

#     dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
#     result = []

#     for i in range(max_length):
#         predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
#         attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

#         predicted_id = tf.argmax(predictions[0]).numpy()
#         result.append (tokenizer.index_word[predicted_id])

#         if tokenizer.index_word[predicted_id] == '<end>':
#             return result, attention_plot,predictions

#         dec_input = tf.expand_dims([predicted_id], 0)

#     attention_plot = attention_plot[:len(result), :]
#     return result, attention_plot,predictions

# def plot_attention_map (caption, weights, image) :

#   fig = plt.figure(figsize = (20, 20))
#   temp_img = np.array(Image.open(image))

#   cap_len = len(caption)
#   for cap in range(cap_len) :
#     weights_img = np.reshape(weights[cap], (8,8))
#     wweights_img = np.array(Image.fromarray(weights_img).resize((224,224), Image.LANCZOS))

#     ax = fig.add_subplot(cap_len//2, cap_len//2, cap+1)
#     ax.set_title(caption[cap], fontsize = 14, color = 'red')

#     img = ax.imshow(temp_img)

#     ax.imshow(weights_img, cmap='gist_heat', alpha=0.6, extent=img.get_extent())
#     ax.axis('off')
#   plt.subplots_adjust(hspace=0.2, wspace=0.2)
#   plt.show()

# from nltk.translate.bleu_score import sentence_bleu

# def filt_text(text):
#     filt=['<start>','<unk>','<end>']
#     temp= text.split()
#     [temp.remove(j) for k in filt for j in temp if k==j]
#     text=' '.join(temp)
#     return text

# image_test = path_test.copy()

# def pred_caption_audio(random, autoplay=False, weights=(0.5, 0.5, 0, 0)) :

#     cap_test_data = caption_test.copy()
#     rid = np.random.randint(0, random)
#     test_image = image_test[rid]
#     real_caption = ' '.join([tokenizer.index_word[i] for i in cap_test_data[rid] if i not in [0]])
#     result, attention_plot, pred_test = evaluate(test_image)
#     real_caption=filt_text(real_caption)
#     pred_caption=' '.join(result).rsplit(' ', 1)[0]
#     real_appn = []
#     real_appn.append(real_caption.split())
#     reference = real_appn
#     candidate = pred_caption.split()
#     score = sentence_bleu(reference, candidate, weights=weights)#set your weights
#     print(f"BLEU score: {score*100}")
#     print ('Real Caption:', real_caption)
#     print ('Prediction Caption:', pred_caption)
#     plot_attention_map(result, attention_plot, test_image)
#     speech = gTTS('Predicted Caption : ' + pred_caption, lang = 'en', slow = False)
#     speech.save('voice.mp3')
#     audio_file = 'voice.mp3'
#     display.display(display.Audio(audio_file, rate = None, autoplay = autoplay))
#     return test_image



# test_image = pred_caption_audio(len(image_test), True, weights = (0.5, 0.25, 0, 0))
# Image.open(test_image)

# test_image = pred_caption_audio(len(image_test), True, weights = (0.5, 0.25, 0, 0))
# Image.open(test_image)