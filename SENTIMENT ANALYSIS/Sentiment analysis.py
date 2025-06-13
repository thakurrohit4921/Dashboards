# %% [markdown]
# # Importing libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from string import punctuation
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, SimpleRNN, LSTM, GRU
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# # Reading our data

# %%
df = pd.read_csv('SENTIMENT ANALYSIS/tripadvisor_hotel_reviews.csv')
df.head()

# %% [markdown]
# # Brief Information about our data

# %%
# No. of rows & Columns and datatypes

df.info()

# %%
# Null Values

df.isnull().sum()

# %% [markdown]
# # Wordcloud

# %%
from wordcloud import WordCloud 
wc = WordCloud(width=800,
               height=500,
               background_color='white',
               min_font_size=10)

# %%
wc.generate(''.join(df['Review']))
plt.figure(figsize=(10,10))
plt.imshow(wc)
plt.axis('off')
plt.show()

# %% [markdown]
# # Data Visualization

# %%
plt.figure(figsize=(10,8))

sns.countplot(data=df,
              x='Rating',
              palette="mako",
              order = df['Rating'].value_counts().index)

plt.title('Rating Distribution')

plt.show()

# %%
plt.pie(df['Rating'].value_counts(), labels=df['Rating'].unique().tolist(), autopct='%1.1f%%')
plt.title('Rating Distribution in percentage')
plt.show()

# %% [markdown]
# # Text Preprocessing 

# %%
# Let's change the rating to be more general and easier to understand
def rating(score):
    if score > 3:
        return 'Good'
    elif score == 3:
        return 'Netral'
    else:
        return 'Bad'

# %%
df['Rating'] = df['Rating'].apply(rating)

# %%
df.head()

# %%
# Creating a Function clean_text for text preprocessing

def clean_text(text):
    stop = stopwords.words('english')       
    punc = list(punctuation)
    bad_tokens = stop + punc
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    word_tokens = [t for t in tokens if t.isalpha()]
    clean_token = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in bad_tokens]
    return " ".join(clean_token)

# %%
# Applying text preprocessing methods to df['Review']

df['Review'] = df['Review'].apply(clean_text)

# %%
df.head()

# %% [markdown]
# # Splitting target and feature columns 

# %%
x = df['Review']
y = df['Rating']

# %%
# Train Test Split

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=1,test_size=0.2)

# %% [markdown]
# # Checking No. of words in each sentence of Review column

# %%
sent_len = []
for sent in df['Review']:
    sent_len.append(len(word_tokenize(sent)))
df['sent_len'] = sent_len
df.head()

# %%
# Checking the reveiw with max no. of words

max(sent_len)

# %%
pd.set_option('display.max_rows', None)
print(df['sent_len'].value_counts())

# %%
# Removing 5% of data beacuse not all sentence length is 39 which is our max sent_len.

np.quantile(sent_len,0.95)

# %%
max_len = 246

# %% [markdown]
# # Tokenization, Vectorization & Padding Sequence

# %%
# Word tokenize on basis of spacing between each words

tok = Tokenizer(char_level=False,split=' ')
tok.fit_on_texts(xtrain)

# %%
# Index of all unique tokenized words 

tok.index_word

# %%
# Total No. of Unique words inour dataset

vocab_len = len(tok.index_word)
vocab_len

# %%
# Putting all index/unique id of each words in a sequence according to the data

sequences_train = tok.texts_to_sequences(xtrain)
sequences_train

# %%
# Padding Sequence

sequence_matrix_train = sequence.pad_sequences(sequences_train,maxlen=max_len)
sequence_matrix_train

# %%
# One hot encoding the label
lb = LabelEncoder()
ytrain = lb.fit_transform(ytrain)
ytest = lb.transform(ytest)

# %% [markdown]
# # Model Building (Neural Network)

# %%
model = Sequential()  
model.add(Embedding(vocab_len+1,500,input_length=max_len,mask_zero=True))                                                  # Embedding
#model.add(SimpleRNN(32,activation='tanh'))                                                                                # RNN Layer
model.add(LSTM(16,activation='tanh'))                                                                                      # LSTM Layer
#model.add(GRU(64,activation='tanh'))                                                                                      # GRU Layer
model.add(Dense(8,activation='relu',kernel_regularizer=regularizers.l2(0.001),bias_regularizer=regularizers.l2(0.001)))    # Hidden Layer
model.add(Dropout(0.5)) 
model.add(Dense(3,activation='softmax'))                                                                                   # Output Layer

# %%
# Summary of our model

model.summary()

# %%
# Compile our Model

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# Training our model

trained_model = model.fit(sequence_matrix_train,ytrain,epochs=20)

# %%
# Giving each unique tokens/word a unique Id or index and the apply padding sequence

sequence_matrix_test = sequence.pad_sequences(tok.texts_to_sequences(xtest),maxlen=max_len)

# %%
# Checking Training and Testing loss 

print('Training_loss :',model.evaluate(sequence_matrix_train,ytrain))
print('Testing_loss :',model.evaluate(sequence_matrix_test,ytest))

# %%
# Testing our model

Y_pred = model.predict(sequence_matrix_test)
print(np.round(Y_pred,3))

# %%
# List comprehension to select class with highest probability

Y_pred = [np.argmax(i) for i in Y_pred]
Y_pred

# %%
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(ytest,Y_pred))
print(confusion_matrix(ytest,Y_pred))

# %% [markdown]
# # Prediction

# %%
def dl_predict(text):
    cleantext = clean_text(text)
    seq = tok.texts_to_sequences([cleantext])
    padded = sequence.pad_sequences(seq)

    pred = model.predict(padded)
    # Get the index of the maximum value in the prediction array
    predicted_index = np.argmax(pred, axis=1)[0]
    # Get the label name using the index
    result = lb.classes_[predicted_index]

    return result

# %%
text = 'Such a comfy place to stay with the loved one'

print('Prediction using DNN: {}'.format(dl_predict(text)))

# %%
text3 = 'Had a bad experience but scenery was good'

print('Prediction using DNN: {}'.format(dl_predict(text3)))


