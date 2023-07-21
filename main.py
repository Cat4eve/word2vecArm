import numpy as np
import tensorflow as tf


## Data Reading 

raw_data = open('corpus_100k', encoding='utf-8').read()

k = 10000 # limited due to low computational resources (you can make more)

splitted_raw_data = raw_data.split(' ')[0:k]

preprocessed_data = np.array([text for text in splitted_raw_data])

m = len(np.unique(preprocessed_data)) # unique words count
 
preprocessed_associative_data = {}
for ind, text in enumerate(np.unique(preprocessed_data)):
    z = np.zeros(m)
    z[ind] = 1
    preprocessed_associative_data[text] = z

preprocessed_data = np.array([preprocessed_associative_data[text] for text in preprocessed_data])

h = 200 # hidden or embedding dimension

X = preprocessed_data[:-1]
y = preprocessed_data[1:]


## Model Creation

inputs = tf.keras.Input(shape=(m))
hiddens = tf.keras.layers.Dense(h, use_bias=False)(inputs)
outputs = tf.keras.layers.Dense(m, use_bias=False, activation=tf.nn.softmax)(hiddens)

word2vecModel = tf.keras.Model(inputs=inputs, outputs=outputs)

word2vecModel.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

word2vecModel.fit(
    x=X,
    y=y,
    epochs=1
)


## Finalizing model

ARMENIAN_EMBEDDED_VECTOR = np.array([
    word2vecModel.layers[0].call(text).reshape(1, -1) @ word2vecModel.layers[1].weights[0] for text in preprocessed_data
])

ARMENIAN_EMBEDDED_VECTOR = ARMENIAN_EMBEDDED_VECTOR.reshape(k, h)

print(ARMENIAN_EMBEDDED_VECTOR[0])



