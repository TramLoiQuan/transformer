from keras.models import Sequential,load_model, Model, model_from_json
from keras.layers import Multiply,Add,Softmax,Permute,Masking,concatenate,Reshape,Conv2D,TimeDistributed,GlobalMaxPool1D,MaxPooling1D,Conv1D,Bidirectional,Input,RepeatVector, Dense,Activation,Concatenate,Dot,LSTM,Lambda,Dropout,Average,Embedding,Flatten,BatchNormalization,Layer
from keras.optimizers import Adam,SGD
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras import regularizers
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import itertools as it
from scipy.sparse import csr_matrix
from sklearn.utils import class_weight
import nltk
from keras.callbacks import ModelCheckpoint,EarlyStopping,Callback
import time

def layer_normalization(x,epsilon=0.001,center=True, scale=True,gamma=1,beta=0):
    mean = K.mean(x, axis=-1, keepdims=True)
    variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
    std = K.sqrt(variance + epsilon)
    outputs = (x - mean) / std
    if scale:
        outputs *= gamma
    if center:
        outputs += beta
    return outputs

class MultiHeadAttention(Layer):
    def __init__(self,max_len,emb_dim,h,**kwargs):
        self.emb_dim = emb_dim
        self.h = h
        self.max_len = max_len
        super(MultiHeadAttention, self).__init__(**kwargs)
  
    def get_config(self):
        config = {
            "max_len": self.max_len,
            "emb_dim": self.emb_dim,
            "h": self.h
        }
        base_config = super(MultiHeadAttention, self).get_config()
        config.update(base_config)
        return config
  
    def build(self, input_shape):
        self.q_dense_layer = Dense(self.emb_dim)
        self.q_dense_layer.build(input_shape[0])

        self.k_dense_layer = Dense(self.emb_dim)
        self.k_dense_layer.build(input_shape[1])

        self.v_dense_layer = Dense(self.emb_dim)
        self.v_dense_layer.build(input_shape[2])

        self.output_dense_layer = Dense(self.emb_dim)
        self.output_dense_layer.build(input_shape[0])

        self._trainable_weights = (self.q_dense_layer.trainable_weights + self.k_dense_layer.trainable_weights 
                                   + self.v_dense_layer.trainable_weights +self.output_dense_layer.trainable_weights)

        super(MultiHeadAttention, self).build(input_shape)
  
    def call(self, x,training=None):
        q = self.q_dense_layer(x[0])
        q = tf.expand_dims(q,axis=2)
        q = tf.split(q,self.h,axis=-1)
        q = tf.concat(q,axis=2)
        q = K.permute_dimensions(q,(0,2,1,3))

        k = self.k_dense_layer(x[1])
        k = tf.expand_dims(k,axis=2)
        k = tf.split(k,self.h,axis=-1)
        k = tf.concat(k,axis=2)
        k = K.permute_dimensions(k,(0,2,1,3))

        v = self.v_dense_layer(x[2])
        v = tf.expand_dims(v,axis=2)
        v = tf.split(v,self.h,axis=-1)
        v = tf.concat(v,axis=2)
        v = K.permute_dimensions(v,(0,2,1,3))

        logits = tf.matmul(q,k,transpose_b=True)

        logits = logits/((self.emb_dim//self.h)**0.5)
        logits = K.softmax(logits,axis=-1)
        attention_output = tf.matmul(logits,v)
        attention_output = K.permute_dimensions(attention_output,(0,2,1,3))
        attention_output = tf.split(attention_output,self.h,axis=2)
        attention_output = tf.concat(attention_output,axis=-1)
        attention_output = tf.squeeze(attention_output,axis=2)
        attention_output = self.output_dense_layer(attention_output)

        return attention_output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],self.max_len,self.emb_dim)

class EncoderAttention(Layer):
    def __init__(self,max_len,emb_dim,h,**kwargs):
        self.emb_dim = emb_dim
        self.h = h
        self.max_len = max_len
        super(EncoderAttention, self).__init__(**kwargs)
  
    def get_config(self):
        config = {
            "max_len": self.max_len,
            "emb_dim": self.emb_dim,
            "h": self.h
        }
        base_config = super(EncoderAttention, self).get_config()
        config.update(base_config)
        return config
  
    def build(self,input_shape):
        self.multihead = MultiHeadAttention(self.max_len,self.emb_dim,self.h)
        self.multihead.build([input_shape,input_shape,input_shape])

        self.conv1 = Conv1D(2048,3,padding='same',activation='relu')
        self.conv1.build((input_shape[0],self.max_len,self.emb_dim))
        
        self.conv2 = Conv1D(self.emb_dim,3,padding='same',activation='relu')
        self.conv2.build((input_shape[0],self.max_len,2048))
        self._trainable_weights = (self.multihead.trainable_weights
                                    + self.conv1.trainable_weights + self.conv2.trainable_weights
                                  )

        super(EncoderAttention, self).build(input_shape)
    
    def call(self, x,training=None):
        multihead_output = self.multihead([x,x,x])
        multihead_output = K.in_train_phase(K.dropout(multihead_output,level=0.2),multihead_output,training=training)
        residual = x + multihead_output
        norm = layer_normalization(residual)
        conv = self.conv1(norm)
        conv = self.conv2(conv)
        residual = norm + conv
        norm = layer_normalization(residual)

        return norm
  
    def compute_output_shape(self, input_shape):
        return input_shape

class DecoderAttention(Layer):
    def __init__(self,max_len,emb_dim,h,**kwargs):
        self.emb_dim = emb_dim
        self.h = h
        self.max_len = max_len
        super(DecoderAttention, self).__init__(**kwargs)
    
    def get_config(self):
        config = {
            "max_len": self.max_len,
            "emb_dim": self.emb_dim,
            "h": self.h
        }
        base_config = super(DecoderAttention, self).get_config()
        config.update(base_config)
        return config
  
    def build(self,input_shape):
        self.multihead1 = MultiHeadAttention(self.max_len,self.emb_dim,self.h)
        self.multihead1.build([input_shape[0],input_shape[0],input_shape[0]])

        self.multihead2 = MultiHeadAttention(self.max_len,self.emb_dim,self.h)
        self.multihead2.build([input_shape[0],input_shape[1],input_shape[1]])

        self.conv1 = Conv1D(2048,3,padding='same',activation='relu')
        self.conv1.build((input_shape[0],self.max_len,self.emb_dim))
        
        self.conv2 = Conv1D(self.emb_dim,3,padding='same',activation='relu')
        self.conv2.build((input_shape[0],self.max_len,2048))
        self._trainable_weights += (self.multihead1.trainable_weights
                                   + self.multihead2.trainable_weights
                                   + self.conv1.trainable_weights +self.conv2.trainable_weights)

        super(DecoderAttention, self).build(input_shape)

    def call(self, x,training=None): ### x[0] = x, x[1] = encoder
        multihead_output = self.multihead1([x[0],x[0],x[0]])
#         multihead_output = K.dropout(multihead_output,level=0.2)
        multihead_output = K.in_train_phase(K.dropout(multihead_output,level=0.2),multihead_output,training=training)
        residual = x[0]+multihead_output
        norm = layer_normalization(residual)

        multihead_output = self.multihead2([norm,x[1],x[1]])
#         multihead_output = K.dropout(multihead_output,level=0.2)
        multihead_output = K.in_train_phase(K.dropout(multihead_output,level=0.2),multihead_output,training=training)
        residual = norm + multihead_output
        norm = layer_normalization(residual)
        conv = self.conv1(norm)
        conv = self.conv2(conv)
        residual = norm+conv
        norm = layer_normalization(residual)

        return norm
  
    def compute_output_shape(self, input_shape):
        return input_shape[0]

def char2idx():
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    alpha_l = "abcdefghijklmnopqrstuvwxyz"
    alpha_u = alpha_l.upper()
    num = "0123456789"
    punc = ",.-:;()\"'"
    vocab = {}
    vocab["#"] = 0  ### zero padding
    vocab['^'] = 1  ### begin
    vocab["$"] = 2  ### end
    vocab[" "] = 3
    for c, i in zip(intab_l+intab_u+alpha_l+alpha_u+num+punc,it.count(4)):
        vocab[c] = i

    return vocab

def idx2char():
    vocab = {}
    for k,v in char2idx().items():
        vocab[v] = k
    return vocab

def generate_batch_input_en(x,vocab,max_len):
    nx = np.zeros((len(x),max_len-1))
    for i, no_tone in enumerate(x):
        for j,c in enumerate(no_tone[:max_len-1]):
            nx[i,j] = vocab[c]
    masking = np.not_equal(nx,0) * 1
    masking = np.expand_dims(masking,axis=-1)
    return nx,masking

def generate_batch_input_de(x,vocab,max_len):
    nx = np.zeros((len(x),max_len))
    for i, no_tone in enumerate(x):
        for j,c in enumerate(("^"+no_tone)[:max_len]):
            nx[i,j] = vocab[c]
    masking = np.not_equal(nx,0) * 1
    masking = np.expand_dims(masking,axis=-1)
    return nx,masking
  
def generate_batch_output(y,vocab,max_len):
    ny = np.zeros((len(y),max_len,len(vocab)))
    for i, tone in enumerate(y):
        for j, c in enumerate((tone+"$")[:max_len]):
            ny[i,j,vocab[c]] = 1
        if len(tone+"$") > max_len:
            ny[i,j,vocab["$"]] = 1
        if j < max_len - 1:
            for n in range(j+1, max_len):
                ny[i,n,0] = 1
    return ny

def position_encoding(max_len,emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
      [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
      if pos != -1 else np.zeros(emb_dim) for pos in range(-1,max_len-1)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) 
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) 
    return position_enc
 
def generate_data(no_tone_path,tone_path,vocab,emb_dim,batch_size,max_len,pos,step):
    x=[]
    y=[]
    i = 0
    k = 0
    while True:
        with open(no_tone_path,'r') as f1, open(tone_path,'r') as f2:
            for l1,l2 in zip(f1,f2):
                if k < step:
                    k+=1
                    continue
                if len(l1.strip()) < max_len-1:
                    x.append(l1.strip())
                    y.append(l2.strip())
                else:
                    for j in nltk.ngrams(l1.strip().split(),max_len//6):
                        x.append(" ".join(j))
                    for j in nltk.ngrams(l2.strip().split(),max_len//6):
                        y.append(" ".join(j))
                i=len(x)
                if i >= batch_size:
                    x_en,x_mask_en = generate_batch_input_en(x,vocab,max_len)
                    x_de,x_mask_de = generate_batch_input_de(x,vocab,max_len)
                    y = generate_batch_output(y,vocab,max_len)
                    broad_pos = np.repeat(np.expand_dims(pos,axis=0),len(x),axis=0)
                    yield ([broad_pos[:,1:,:],x_en,x_mask_en,broad_pos,x_de,x_mask_de],y)
                    x=[]
                    y=[]
                    i = 0
        if i!=0:
            x_en,x_mask_en = generate_batch_input_en(x,vocab,max_len)
            x_de,x_mask_de = generate_batch_input_de(x,vocab,max_len)
            y = generate_batch_output(y,vocab,max_len)
            broad_pos = np.repeat(np.expand_dims(pos,axis=0),len(x),axis=0)
            yield ([broad_pos[:,1:,:],x_en,x_mask_en,broad_pos,x_de,x_mask_de],y)
            x=[]
            y=[]
            i = 0
            
class TimerCallback(Callback):
    
    def __init__(self, maxExecutionTime, byBatch = False, on_interrupt=None):
        
        # Arguments:
        #     maxExecutionTime (number): Time in minutes. The model will keep training 
        #                                until shortly before this limit
        #                                (If you need safety, provide a time with a certain tolerance)

        #     byBatch (boolean)     : If True, will try to interrupt training at the end of each batch
        #                             If False, will try to interrupt the model at the end of each epoch    
        #                            (use `byBatch = True` only if each epoch is going to take hours)          

        #     on_interrupt (method)          : called when training is interrupted
        #         signature: func(model,elapsedTime), where...
        #               model: the model being trained
        #               elapsedTime: the time passed since the beginning until interruption   
        #the same handler is used for checking each batch or each epoch
        self.maxExecutionTime = maxExecutionTime * 60
        self.on_interrupt = on_interrupt
        if byBatch == True:
            #on_batch_end is called by keras every time a batch finishes
            self.on_batch_end = self.on_end_handler
        else:
            #on_epoch_end is called by keras every time an epoch finishes
            self.on_epoch_end = self.on_end_handler
    
    #Keras will call this when training begins
    def on_train_begin(self, logs):
        self.startTime = time.time()
        self.longestTime = 0            #time taken by the longest epoch or batch
        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished

        #this is our custom handler that will be used in place of the keras methods:
            #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`
        
    def on_end_handler(self, index, logs):
        
        currentTime      = time.time()                           
        self.elapsedTime = currentTime - self.startTime    #total time taken until now
        thisTime         = currentTime - self.lastTime     #time taken for the current epoch
                                                               #or batch to finish
        self.lastTime = currentTime
        
        #verifications will be made based on the longest epoch or batch
        if thisTime > self.longestTime:
            self.longestTime = thisTime
        
        #if the (assumed) time taken by the next epoch or batch is greater than the
            #remaining time, stop training
        remainingTime = self.maxExecutionTime - self.elapsedTime
        if remainingTime < self.longestTime:
            self.model.stop_training = True  #this tells Keras to not continue training
            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")
        
        #if we have passed the `on_interrupt` callback, call it here
            if self.on_interrupt is not None:
                self.on_interrupt(self.model, self.elapsedTime)

class Transformer():
    def __init__(self,max_len,emb_dim,vocab_len,h=8):
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.h = h
        self.vocab_len = vocab_len
        self.embedding = Embedding(input_dim=self.vocab_len, output_dim=self.emb_dim)

        self.decoder = [DecoderAttention(self.max_len,self.emb_dim,self.h),
                       DecoderAttention(self.max_len,self.emb_dim,self.h),
                       DecoderAttention(self.max_len,self.emb_dim,self.h),
                        DecoderAttention(self.max_len,self.emb_dim,self.h),
                        DecoderAttention(self.max_len,self.emb_dim,self.h),
                        DecoderAttention(self.max_len,self.emb_dim,self.h),
                       Dense(self.vocab_len,activation='softmax')]

        self.encoder = [EncoderAttention(self.max_len-1,self.emb_dim,self.h),
                       EncoderAttention(self.max_len-1,self.emb_dim,self.h),
                        EncoderAttention(self.max_len-1,self.emb_dim,self.h),
                        EncoderAttention(self.max_len-1,self.emb_dim,self.h),
                        EncoderAttention(self.max_len-1,self.emb_dim,self.h),
                       EncoderAttention(self.max_len-1,self.emb_dim,self.h)]

        self.build_model()
#         self.inference_model()
    
    def character_position_embeddings(self,sent_input,masking_input,pos_encode):
        embeds = self.embedding(sent_input)
        embeds = Add()([embeds,pos_encode])
        embeds = Multiply()([embeds, masking_input])
        return embeds

    def encoder_stack(self,x):
        en = x
        for en_layer in self.encoder:
            en = en_layer(en)

        return en

    def decoder_stack(self,x, encoder):
        de = x
        for de_layer in self.decoder[:-1]:
            de = de_layer([de,encoder])

        de = self.decoder[-1](de)

        return de
  
    def build_model(self):
                   
        pos_emb_encoder = Input(shape=(self.max_len-1,self.emb_dim))          
        sent_input = Input(shape=(self.max_len-1,))
        en_masking_input = Input(shape=(self.max_len-1,1))

        en_char_embeds = self.character_position_embeddings(sent_input,en_masking_input,pos_emb_encoder)
        en = self.encoder_stack(en_char_embeds) 

        pos_emb_decoder = Input(shape=(self.max_len,self.emb_dim))
        decode_input = Input(shape=(self.max_len,))
        de_masking_input = Input(shape=(self.max_len,1))

        de_char_embeds = self.character_position_embeddings(decode_input,de_masking_input,pos_emb_decoder)
        de = self.decoder_stack(de_char_embeds,en)

        self.model = Model([pos_emb_encoder,sent_input,en_masking_input,pos_emb_decoder,decode_input,de_masking_input],de)
        opt = Adam(lr=0.00001,beta_1=0.9,beta_2=0.99,epsilon=10**-9,decay=0.01)
        ### 0.00006
        self.model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    
    def train_on_generator(self,no_tone_path,tone_path,no_tone_path_val,tone_path_val,vocab,train_size,batch_size,pos,epochs,step):
        train_generator = generate_data(no_tone_path,tone_path,vocab,self.emb_dim,batch_size,self.max_len,pos,step)
        validate_generator = generate_data(no_tone_path_val,tone_path_val,vocab,self.emb_dim,batch_size,self.max_len,pos,0)

#         self.model.fit([np.repeat(np.expand_dims(pos_encoding[1:],axis=0),len(x),axis=0),x_train,x_masking,np.repeat(np.expand_dims(pos_encoding,axis=0),len(x),axis=0),y_train,y_masking],label,validation_split=0.05)
#         self.model.train_on_batch([np.repeat(np.expand_dims(pos_encoding,axis=0),len(x),axis=0),x_train,x_masking,np.repeat(np.expand_dims(pos_encoding,axis=0),len(x),axis=0),y_train,y_masking],label)
        
        checkpointer = ModelCheckpoint(filepath=os.path.join('./transformer_{val_loss:.5f}_{val_acc:.5f}.h5'), save_best_only=True, verbose=1)
#         early = EarlyStopping(patience=2, verbose=1)
        timerCallback = TimerCallback(350,True)
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=train_size//batch_size,
                                 validation_data=validate_generator,
                                 validation_steps=230000//batch_size,
                                 epochs=epochs,
                                 callbacks=[checkpointer])
    
    def evaluate_generator(self,no_tone_path_val,tone_path_val,vocab,batch_size,pos):
        validate_generator = generate_data(no_tone_path_val,tone_path_val,vocab,self.emb_dim,batch_size,self.max_len,pos)
        
        print(self.model.evaluate_generator(validate_generator,steps=1,verbose=1))
        
    def predict(self,x,pos_encoding,vocab,inverse_vocab):
        lw = x.split()
#         ls = [" ".join(lw[i*6:(i+1)*6]) for i in range(len(lw)//6+1)]
        ls = [" ".join(i) for i in list(nltk.ngrams(lw,10))]
        if len(ls) == 0:
            ls = [x]
        x_p1, x_masking1 = generate_batch_input_en(ls,vocab,self.max_len)
        x_p2, x_masking2 = generate_batch_input_de(ls,vocab,self.max_len)
        broad_pos_en = np.repeat(np.expand_dims(pos_encoding,axis=0),len(x_p1),axis=0)
        
        de_output = self.model.predict([broad_pos_en[:,1:,:],x_p1,x_masking1,broad_pos_en,x_p2,x_masking2])
        de_output = np.argmax(de_output,axis=-1)
        
        def to_string(lidx):
            s = ""
            for w in lidx:
                s+=inverse_vocab[w]
            return s
        l = []
        for d in de_output:
            l.append(to_string(d).split())
            
        mod = (len(l)-1)%3
        al = l[0::3]

        s = " ".join(al[0][0:6])
        for i in range(1,len(al)):
            s += " "+" ".join(al[i][3:6])
        # print(l[-1])
        if mod == 0:
            s+= " " +" ".join(l[-1][6:])
        elif mod == 1:
            s+= " " +" ".join(l[-1][5:])
        else: 
            s+= " " +" ".join(l[-1][4:])
        return s[:len(x)]
    
    def save(self,step,path="transformer1"):
        self.model.save_weights(path+"{:02d}.h5".format(step))
        print("SAVE MODEL STEP: ",step)
        
    def load(self,step,path="transformer1"):
        self.model.load_weights(path+"{:02d}.h5".format(step))