# XOR GATE IMPLEMENTATION

## Aim:
To implement multi layer artificial neural network using back propagation algorithm.


## Equipments Required:
Hardware – PCs

Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab


## Related Theory Concept:
Logic gates are neural networks help to understand the mathematical computation by which a neural network processes its input s to achieve at a certain output. This neural network will deal with the XOR logic problem. An XOR (exclusive OR gate) is a digital logic gate that gives a true output only when both its inputs differ from each other. The information of a neural network is stored in the interconnections between the neurons i.e. the weights. A neural network learns by updating its weights according to a learning algorithm that helps it converge to the expected output .The learning algorithm is a principled way of changing the weights and biases based on the loss function.

## Algorithm
1. Import the required libraries.

2. Create the training dataset.

3. Create the neural network model with one hidden layer.

4. Train the model with training data.

5. Now test the model with testing data.

## Program
```

/*
Program to implement XOR Logic Gate.
Developed by   :  S. Sanju
RegisterNumber :  212219040137
*/

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data=np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
target_data=np.array([[0],[1],[1],[0]],"float32")

model=Sequential()
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mean_squared_error',
                    optimizer='adam',
                    metrics=['binary_accuracy'])
model.fit(training_data,target_data,epochs=1000)
scores=model.evaluate(training_data,target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
print(model.predict(training_data).round())

```

## Output
```
Epoch 1/1000
1/1 [==============================] - 0s 184ms/step - loss: 0.2525 - binary_accuracy: 0.7500
Epoch 2/1000
1/1 [==============================] - 0s 17ms/step - loss: 0.2521 - binary_accuracy: 0.7500
Epoch 3/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2517 - binary_accuracy: 0.7500
Epoch 4/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2512 - binary_accuracy: 0.7500
Epoch 5/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2508 - binary_accuracy: 0.7500
Epoch 6/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2503 - binary_accuracy: 0.7500
Epoch 7/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2499 - binary_accuracy: 0.7500
Epoch 8/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2495 - binary_accuracy: 0.7500
Epoch 9/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2490 - binary_accuracy: 0.7500
Epoch 10/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2486 - binary_accuracy: 0.7500
Epoch 11/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2482 - binary_accuracy: 0.7500
Epoch 12/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2478 - binary_accuracy: 0.7500
Epoch 13/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2474 - binary_accuracy: 0.7500
Epoch 14/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2470 - binary_accuracy: 0.7500
Epoch 15/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.2466 - binary_accuracy: 0.7500
Epoch 16/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2462 - binary_accuracy: 0.5000
Epoch 17/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2458 - binary_accuracy: 0.5000
Epoch 18/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.2454 - binary_accuracy: 0.5000
Epoch 19/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2451 - binary_accuracy: 0.5000
Epoch 20/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.2447 - binary_accuracy: 0.7500
Epoch 21/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2443 - binary_accuracy: 0.7500
Epoch 22/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2440 - binary_accuracy: 0.7500
Epoch 23/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2436 - binary_accuracy: 0.7500
Epoch 24/1000
1/1 [==============================] - 0s 16ms/step - loss: 0.2433 - binary_accuracy: 0.7500
Epoch 25/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2430 - binary_accuracy: 0.7500
Epoch 26/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2426 - binary_accuracy: 0.7500
Epoch 27/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2423 - binary_accuracy: 0.7500
Epoch 28/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2419 - binary_accuracy: 0.7500
Epoch 29/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2416 - binary_accuracy: 0.7500
Epoch 30/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2412 - binary_accuracy: 0.7500
Epoch 31/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2409 - binary_accuracy: 0.7500
Epoch 32/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2406 - binary_accuracy: 0.7500
Epoch 33/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2402 - binary_accuracy: 0.7500
Epoch 34/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2399 - binary_accuracy: 1.0000
Epoch 35/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2395 - binary_accuracy: 1.0000
Epoch 36/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2392 - binary_accuracy: 1.0000
Epoch 37/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2389 - binary_accuracy: 1.0000
Epoch 38/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2385 - binary_accuracy: 1.0000
Epoch 39/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2382 - binary_accuracy: 1.0000
Epoch 40/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2379 - binary_accuracy: 1.0000
Epoch 41/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2375 - binary_accuracy: 1.0000
Epoch 42/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2372 - binary_accuracy: 1.0000
Epoch 43/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.2368 - binary_accuracy: 1.0000
Epoch 44/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2365 - binary_accuracy: 1.0000
Epoch 45/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2362 - binary_accuracy: 1.0000
Epoch 46/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2358 - binary_accuracy: 1.0000
Epoch 47/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2355 - binary_accuracy: 1.0000
Epoch 48/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2352 - binary_accuracy: 1.0000
Epoch 49/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2349 - binary_accuracy: 1.0000
Epoch 50/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2346 - binary_accuracy: 1.0000
Epoch 51/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2344 - binary_accuracy: 1.0000
Epoch 52/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2341 - binary_accuracy: 1.0000
Epoch 53/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2338 - binary_accuracy: 1.0000
Epoch 54/1000
1/1 [==============================] - 0s 500us/step - loss: 0.2335 - binary_accuracy: 1.0000
Epoch 55/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2332 - binary_accuracy: 1.0000
Epoch 56/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2330 - binary_accuracy: 1.0000
Epoch 57/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2327 - binary_accuracy: 1.0000
Epoch 58/1000
1/1 [==============================] - 0s 15ms/step - loss: 0.2324 - binary_accuracy: 1.0000
Epoch 59/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2321 - binary_accuracy: 1.0000
Epoch 60/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2318 - binary_accuracy: 1.0000
Epoch 61/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2316 - binary_accuracy: 1.0000
Epoch 62/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2313 - binary_accuracy: 1.0000
Epoch 63/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2311 - binary_accuracy: 1.0000
Epoch 64/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2308 - binary_accuracy: 1.0000
Epoch 65/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2306 - binary_accuracy: 1.0000
Epoch 66/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2303 - binary_accuracy: 1.0000
Epoch 67/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2301 - binary_accuracy: 1.0000
Epoch 68/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2298 - binary_accuracy: 1.0000
Epoch 69/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2296 - binary_accuracy: 1.0000
Epoch 70/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2293 - binary_accuracy: 1.0000
Epoch 71/1000
1/1 [==============================] - 0s 16ms/step - loss: 0.2291 - binary_accuracy: 1.0000
Epoch 72/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2288 - binary_accuracy: 1.0000
Epoch 73/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2286 - binary_accuracy: 1.0000
Epoch 74/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2283 - binary_accuracy: 1.0000
Epoch 75/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2281 - binary_accuracy: 1.0000
Epoch 76/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2278 - binary_accuracy: 1.0000
Epoch 77/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2275 - binary_accuracy: 1.0000
Epoch 78/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2273 - binary_accuracy: 1.0000
Epoch 79/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2270 - binary_accuracy: 1.0000
Epoch 80/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2268 - binary_accuracy: 1.0000
Epoch 81/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2265 - binary_accuracy: 1.0000
Epoch 82/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2262 - binary_accuracy: 1.0000
Epoch 83/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2260 - binary_accuracy: 1.0000
Epoch 84/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2257 - binary_accuracy: 1.0000
Epoch 85/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2254 - binary_accuracy: 1.0000
Epoch 86/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.2252 - binary_accuracy: 1.0000
Epoch 87/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2249 - binary_accuracy: 1.0000
Epoch 88/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2246 - binary_accuracy: 1.0000
Epoch 89/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2244 - binary_accuracy: 1.0000
Epoch 90/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2241 - binary_accuracy: 1.0000
Epoch 91/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.2238 - binary_accuracy: 1.0000
Epoch 92/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2236 - binary_accuracy: 1.0000
Epoch 93/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2233 - binary_accuracy: 1.0000
Epoch 94/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2230 - binary_accuracy: 1.0000
Epoch 95/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2227 - binary_accuracy: 1.0000
Epoch 96/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2225 - binary_accuracy: 1.0000
Epoch 97/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2222 - binary_accuracy: 1.0000
Epoch 98/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2219 - binary_accuracy: 1.0000
Epoch 99/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.2216 - binary_accuracy: 1.0000
Epoch 100/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2213 - binary_accuracy: 1.0000
Epoch 101/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2211 - binary_accuracy: 1.0000
Epoch 102/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.2208 - binary_accuracy: 1.0000
Epoch 103/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2205 - binary_accuracy: 1.0000
Epoch 104/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2202 - binary_accuracy: 1.0000
Epoch 105/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2199 - binary_accuracy: 1.0000
Epoch 106/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2196 - binary_accuracy: 1.0000
Epoch 107/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2194 - binary_accuracy: 1.0000
Epoch 108/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2191 - binary_accuracy: 1.0000
Epoch 109/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2188 - binary_accuracy: 1.0000
Epoch 110/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2185 - binary_accuracy: 1.0000
Epoch 111/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2182 - binary_accuracy: 1.0000
Epoch 112/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2179 - binary_accuracy: 1.0000
Epoch 113/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2176 - binary_accuracy: 1.0000
Epoch 114/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2173 - binary_accuracy: 1.0000
Epoch 115/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.2170 - binary_accuracy: 1.0000
Epoch 116/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2167 - binary_accuracy: 1.0000
Epoch 117/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2164 - binary_accuracy: 1.0000
Epoch 118/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2161 - binary_accuracy: 1.0000
Epoch 119/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2159 - binary_accuracy: 1.0000
Epoch 120/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2156 - binary_accuracy: 1.0000
Epoch 121/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.2154 - binary_accuracy: 1.0000
Epoch 122/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2151 - binary_accuracy: 1.0000
Epoch 123/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.2148 - binary_accuracy: 1.0000
Epoch 124/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2146 - binary_accuracy: 1.0000
Epoch 125/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2143 - binary_accuracy: 1.0000
Epoch 126/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.2140 - binary_accuracy: 1.0000
Epoch 127/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2138 - binary_accuracy: 1.0000
Epoch 128/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2135 - binary_accuracy: 1.0000
Epoch 129/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2132 - binary_accuracy: 1.0000
Epoch 130/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2130 - binary_accuracy: 1.0000
Epoch 131/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2127 - binary_accuracy: 1.0000
Epoch 132/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2124 - binary_accuracy: 1.0000
Epoch 133/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2121 - binary_accuracy: 1.0000
Epoch 134/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2119 - binary_accuracy: 1.0000
Epoch 135/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.2116 - binary_accuracy: 1.0000
Epoch 136/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2113 - binary_accuracy: 1.0000
Epoch 137/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2110 - binary_accuracy: 1.0000
Epoch 138/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2107 - binary_accuracy: 1.0000
Epoch 139/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2105 - binary_accuracy: 1.0000
Epoch 140/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2102 - binary_accuracy: 1.0000
Epoch 141/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2099 - binary_accuracy: 1.0000
Epoch 142/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2096 - binary_accuracy: 1.0000
Epoch 143/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2093 - binary_accuracy: 1.0000
Epoch 144/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2090 - binary_accuracy: 1.0000
Epoch 145/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2087 - binary_accuracy: 1.0000
Epoch 146/1000
1/1 [==============================] - 0s 469us/step - loss: 0.2085 - binary_accuracy: 1.0000
Epoch 147/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2082 - binary_accuracy: 1.0000
Epoch 148/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2079 - binary_accuracy: 1.0000
Epoch 149/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2076 - binary_accuracy: 1.0000
Epoch 150/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2073 - binary_accuracy: 1.0000
Epoch 151/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2070 - binary_accuracy: 1.0000
Epoch 152/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.2067 - binary_accuracy: 1.0000
Epoch 153/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2064 - binary_accuracy: 1.0000
Epoch 154/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2061 - binary_accuracy: 1.0000
Epoch 155/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2058 - binary_accuracy: 1.0000
Epoch 156/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2055 - binary_accuracy: 1.0000
Epoch 157/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2052 - binary_accuracy: 1.0000
Epoch 158/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2049 - binary_accuracy: 1.0000
Epoch 159/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2046 - binary_accuracy: 1.0000
Epoch 160/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.2043 - binary_accuracy: 1.0000
Epoch 161/1000
1/1 [==============================] - 0s 0s/step - loss: 0.2040 - binary_accuracy: 1.0000
Epoch 162/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.2037 - binary_accuracy: 1.0000
Epoch 163/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2034 - binary_accuracy: 1.0000
Epoch 164/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2031 - binary_accuracy: 1.0000
Epoch 165/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2028 - binary_accuracy: 1.0000
Epoch 166/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.2025 - binary_accuracy: 1.0000
Epoch 167/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2022 - binary_accuracy: 1.0000
Epoch 168/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2019 - binary_accuracy: 1.0000
Epoch 169/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2016 - binary_accuracy: 1.0000
Epoch 170/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2013 - binary_accuracy: 1.0000
Epoch 171/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.2010 - binary_accuracy: 1.0000
Epoch 172/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.2006 - binary_accuracy: 1.0000
Epoch 173/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2003 - binary_accuracy: 1.0000
Epoch 174/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.2000 - binary_accuracy: 1.0000
Epoch 175/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1997 - binary_accuracy: 1.0000
Epoch 176/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1994 - binary_accuracy: 1.0000
Epoch 177/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1991 - binary_accuracy: 1.0000
Epoch 178/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1988 - binary_accuracy: 1.0000
Epoch 179/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1984 - binary_accuracy: 1.0000
Epoch 180/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1981 - binary_accuracy: 1.0000
Epoch 181/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1978 - binary_accuracy: 1.0000
Epoch 182/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1975 - binary_accuracy: 1.0000
Epoch 183/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.1972 - binary_accuracy: 1.0000
Epoch 184/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1969 - binary_accuracy: 1.0000
Epoch 185/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1965 - binary_accuracy: 1.0000
Epoch 186/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1962 - binary_accuracy: 1.0000
Epoch 187/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1959 - binary_accuracy: 1.0000
Epoch 188/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1956 - binary_accuracy: 1.0000
Epoch 189/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1952 - binary_accuracy: 1.0000
Epoch 190/1000
1/1 [==============================] - 0s 15ms/step - loss: 0.1949 - binary_accuracy: 1.0000
Epoch 191/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1946 - binary_accuracy: 1.0000
Epoch 192/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1943 - binary_accuracy: 1.0000
Epoch 193/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1939 - binary_accuracy: 1.0000
Epoch 194/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1936 - binary_accuracy: 1.0000
Epoch 195/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.1933 - binary_accuracy: 1.0000
Epoch 196/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1929 - binary_accuracy: 1.0000
Epoch 197/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1926 - binary_accuracy: 1.0000
Epoch 198/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1923 - binary_accuracy: 1.0000
Epoch 199/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1919 - binary_accuracy: 1.0000
Epoch 200/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1916 - binary_accuracy: 1.0000
Epoch 201/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1913 - binary_accuracy: 1.0000
Epoch 202/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1909 - binary_accuracy: 1.0000
Epoch 203/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1906 - binary_accuracy: 1.0000
Epoch 204/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1902 - binary_accuracy: 1.0000
Epoch 205/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1899 - binary_accuracy: 1.0000
Epoch 206/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1896 - binary_accuracy: 1.0000
Epoch 207/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1892 - binary_accuracy: 1.0000
Epoch 208/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1889 - binary_accuracy: 1.0000
Epoch 209/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1885 - binary_accuracy: 1.0000
Epoch 210/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1882 - binary_accuracy: 1.0000
Epoch 211/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1879 - binary_accuracy: 1.0000
Epoch 212/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1875 - binary_accuracy: 1.0000
Epoch 213/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1872 - binary_accuracy: 1.0000
Epoch 214/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1868 - binary_accuracy: 1.0000
Epoch 215/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1865 - binary_accuracy: 1.0000
Epoch 216/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1861 - binary_accuracy: 1.0000
Epoch 217/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1858 - binary_accuracy: 1.0000
Epoch 218/1000
1/1 [==============================] - 0s 13ms/step - loss: 0.1854 - binary_accuracy: 1.0000
Epoch 219/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1851 - binary_accuracy: 1.0000
Epoch 220/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1847 - binary_accuracy: 1.0000
Epoch 221/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1844 - binary_accuracy: 1.0000
Epoch 222/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1840 - binary_accuracy: 1.0000
Epoch 223/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1837 - binary_accuracy: 1.0000
Epoch 224/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1833 - binary_accuracy: 1.0000
Epoch 225/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1830 - binary_accuracy: 1.0000
Epoch 226/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1826 - binary_accuracy: 1.0000
Epoch 227/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1823 - binary_accuracy: 1.0000
Epoch 228/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1819 - binary_accuracy: 1.0000
Epoch 229/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1815 - binary_accuracy: 1.0000
Epoch 230/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1812 - binary_accuracy: 1.0000
Epoch 231/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1808 - binary_accuracy: 1.0000
Epoch 232/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1805 - binary_accuracy: 1.0000
Epoch 233/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1801 - binary_accuracy: 1.0000
Epoch 234/1000
1/1 [==============================] - 0s 16ms/step - loss: 0.1797 - binary_accuracy: 1.0000
Epoch 235/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1794 - binary_accuracy: 1.0000
Epoch 236/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1790 - binary_accuracy: 1.0000
Epoch 237/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1787 - binary_accuracy: 1.0000
Epoch 238/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1783 - binary_accuracy: 1.0000
Epoch 239/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1779 - binary_accuracy: 1.0000
Epoch 240/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1776 - binary_accuracy: 1.0000
Epoch 241/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1772 - binary_accuracy: 1.0000
Epoch 242/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1769 - binary_accuracy: 1.0000
Epoch 243/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1765 - binary_accuracy: 1.0000
Epoch 244/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1761 - binary_accuracy: 1.0000
Epoch 245/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1758 - binary_accuracy: 1.0000
Epoch 246/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1754 - binary_accuracy: 1.0000
Epoch 247/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1750 - binary_accuracy: 1.0000
Epoch 248/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1747 - binary_accuracy: 1.0000
Epoch 249/1000
1/1 [==============================] - 0s 16ms/step - loss: 0.1743 - binary_accuracy: 1.0000
Epoch 250/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1739 - binary_accuracy: 1.0000
Epoch 251/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1736 - binary_accuracy: 1.0000
Epoch 252/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.1732 - binary_accuracy: 1.0000
Epoch 253/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1728 - binary_accuracy: 1.0000
Epoch 254/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1724 - binary_accuracy: 1.0000
Epoch 255/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1721 - binary_accuracy: 1.0000
Epoch 256/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1717 - binary_accuracy: 1.0000
Epoch 257/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1713 - binary_accuracy: 1.0000
Epoch 258/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1710 - binary_accuracy: 1.0000
Epoch 259/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1706 - binary_accuracy: 1.0000
Epoch 260/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1702 - binary_accuracy: 1.0000
Epoch 261/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1698 - binary_accuracy: 1.0000
Epoch 262/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1695 - binary_accuracy: 1.0000
Epoch 263/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1691 - binary_accuracy: 1.0000
Epoch 264/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1687 - binary_accuracy: 1.0000
Epoch 265/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1684 - binary_accuracy: 1.0000
Epoch 266/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1680 - binary_accuracy: 1.0000
Epoch 267/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1676 - binary_accuracy: 1.0000
Epoch 268/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1673 - binary_accuracy: 1.0000
Epoch 269/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1669 - binary_accuracy: 1.0000
Epoch 270/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1665 - binary_accuracy: 1.0000
Epoch 271/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1662 - binary_accuracy: 1.0000
Epoch 272/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1658 - binary_accuracy: 1.0000
Epoch 273/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1654 - binary_accuracy: 1.0000
Epoch 274/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1651 - binary_accuracy: 1.0000
Epoch 275/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.1647 - binary_accuracy: 1.0000
Epoch 276/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1643 - binary_accuracy: 1.0000
Epoch 277/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1640 - binary_accuracy: 1.0000
Epoch 278/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.1636 - binary_accuracy: 1.0000
Epoch 279/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1632 - binary_accuracy: 1.0000
Epoch 280/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1629 - binary_accuracy: 1.0000
Epoch 281/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.1625 - binary_accuracy: 1.0000
Epoch 282/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1621 - binary_accuracy: 1.0000
Epoch 283/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1618 - binary_accuracy: 1.0000
Epoch 284/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1614 - binary_accuracy: 1.0000
Epoch 285/1000
1/1 [==============================] - 0s 14ms/step - loss: 0.1610 - binary_accuracy: 1.0000
Epoch 286/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1606 - binary_accuracy: 1.0000
Epoch 287/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1603 - binary_accuracy: 1.0000
Epoch 288/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1599 - binary_accuracy: 1.0000
Epoch 289/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1595 - binary_accuracy: 1.0000
Epoch 290/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1592 - binary_accuracy: 1.0000
Epoch 291/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1588 - binary_accuracy: 1.0000
Epoch 292/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1584 - binary_accuracy: 1.0000
Epoch 293/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1581 - binary_accuracy: 1.0000
Epoch 294/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1577 - binary_accuracy: 1.0000
Epoch 295/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1573 - binary_accuracy: 1.0000
Epoch 296/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1570 - binary_accuracy: 1.0000
Epoch 297/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1566 - binary_accuracy: 1.0000
Epoch 298/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1563 - binary_accuracy: 1.0000
Epoch 299/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.1559 - binary_accuracy: 1.0000
Epoch 300/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1555 - binary_accuracy: 1.0000
Epoch 301/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1551 - binary_accuracy: 1.0000
Epoch 302/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1548 - binary_accuracy: 1.0000
Epoch 303/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1544 - binary_accuracy: 1.0000
Epoch 304/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1540 - binary_accuracy: 1.0000
Epoch 305/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.1537 - binary_accuracy: 1.0000
Epoch 306/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1533 - binary_accuracy: 1.0000
Epoch 307/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1529 - binary_accuracy: 1.0000
Epoch 308/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1526 - binary_accuracy: 1.0000
Epoch 309/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1522 - binary_accuracy: 1.0000
Epoch 310/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1518 - binary_accuracy: 1.0000
Epoch 311/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.1515 - binary_accuracy: 1.0000
Epoch 312/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1511 - binary_accuracy: 1.0000
Epoch 313/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1507 - binary_accuracy: 1.0000
Epoch 314/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1504 - binary_accuracy: 1.0000
Epoch 315/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1500 - binary_accuracy: 1.0000
Epoch 316/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1496 - binary_accuracy: 1.0000
Epoch 317/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1493 - binary_accuracy: 1.0000
Epoch 318/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1489 - binary_accuracy: 1.0000
Epoch 319/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1485 - binary_accuracy: 1.0000
Epoch 320/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1482 - binary_accuracy: 1.0000
Epoch 321/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1478 - binary_accuracy: 1.0000
Epoch 322/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1475 - binary_accuracy: 1.0000
Epoch 323/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1471 - binary_accuracy: 1.0000
Epoch 324/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1468 - binary_accuracy: 1.0000
Epoch 325/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1464 - binary_accuracy: 1.0000
Epoch 326/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1460 - binary_accuracy: 1.0000
Epoch 327/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1457 - binary_accuracy: 1.0000
Epoch 328/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1453 - binary_accuracy: 1.0000
Epoch 329/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1450 - binary_accuracy: 1.0000
Epoch 330/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1446 - binary_accuracy: 1.0000
Epoch 331/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1443 - binary_accuracy: 1.0000
Epoch 332/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.1439 - binary_accuracy: 1.0000
Epoch 333/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1435 - binary_accuracy: 1.0000
Epoch 334/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1432 - binary_accuracy: 1.0000
Epoch 335/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1428 - binary_accuracy: 1.0000
Epoch 336/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1425 - binary_accuracy: 1.0000
Epoch 337/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.1421 - binary_accuracy: 1.0000
Epoch 338/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1417 - binary_accuracy: 1.0000
Epoch 339/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1414 - binary_accuracy: 1.0000
Epoch 340/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1411 - binary_accuracy: 1.0000
Epoch 341/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1407 - binary_accuracy: 1.0000
Epoch 342/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.1403 - binary_accuracy: 1.0000
Epoch 343/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1400 - binary_accuracy: 1.0000
Epoch 344/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1396 - binary_accuracy: 1.0000
Epoch 345/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1393 - binary_accuracy: 1.0000
Epoch 346/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1390 - binary_accuracy: 1.0000
Epoch 347/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1386 - binary_accuracy: 1.0000
Epoch 348/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1382 - binary_accuracy: 1.0000
Epoch 349/1000
1/1 [==============================] - 0s 18ms/step - loss: 0.1379 - binary_accuracy: 1.0000
Epoch 350/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1375 - binary_accuracy: 1.0000
Epoch 351/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1372 - binary_accuracy: 1.0000
Epoch 352/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1368 - binary_accuracy: 1.0000
Epoch 353/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1365 - binary_accuracy: 1.0000
Epoch 354/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1361 - binary_accuracy: 1.0000
Epoch 355/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1358 - binary_accuracy: 1.0000
Epoch 356/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1355 - binary_accuracy: 1.0000
Epoch 357/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.1351 - binary_accuracy: 1.0000
Epoch 358/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1348 - binary_accuracy: 1.0000
Epoch 359/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1344 - binary_accuracy: 1.0000
Epoch 360/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1340 - binary_accuracy: 1.0000
Epoch 361/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1337 - binary_accuracy: 1.0000
Epoch 362/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1334 - binary_accuracy: 1.0000
Epoch 363/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1330 - binary_accuracy: 1.0000
Epoch 364/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1327 - binary_accuracy: 1.0000
Epoch 365/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1323 - binary_accuracy: 1.0000
Epoch 366/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1320 - binary_accuracy: 1.0000
Epoch 367/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1317 - binary_accuracy: 1.0000
Epoch 368/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1313 - binary_accuracy: 1.0000
Epoch 369/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1310 - binary_accuracy: 1.0000
Epoch 370/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1306 - binary_accuracy: 1.0000
Epoch 371/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1303 - binary_accuracy: 1.0000
Epoch 372/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.1300 - binary_accuracy: 1.0000
Epoch 373/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1296 - binary_accuracy: 1.0000
Epoch 374/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1293 - binary_accuracy: 1.0000
Epoch 375/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1289 - binary_accuracy: 1.0000
Epoch 376/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1286 - binary_accuracy: 1.0000
Epoch 377/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1283 - binary_accuracy: 1.0000
Epoch 378/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1279 - binary_accuracy: 1.0000
Epoch 379/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1276 - binary_accuracy: 1.0000
Epoch 380/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1272 - binary_accuracy: 1.0000
Epoch 381/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1269 - binary_accuracy: 1.0000
Epoch 382/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1266 - binary_accuracy: 1.0000
Epoch 383/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1262 - binary_accuracy: 1.0000
Epoch 384/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.1259 - binary_accuracy: 1.0000
Epoch 385/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1256 - binary_accuracy: 1.0000
Epoch 386/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1252 - binary_accuracy: 1.0000
Epoch 387/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1249 - binary_accuracy: 1.0000
Epoch 388/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1246 - binary_accuracy: 1.0000
Epoch 389/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1242 - binary_accuracy: 1.0000
Epoch 390/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1239 - binary_accuracy: 1.0000
Epoch 391/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1236 - binary_accuracy: 1.0000
Epoch 392/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1232 - binary_accuracy: 1.0000
Epoch 393/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1229 - binary_accuracy: 1.0000
Epoch 394/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1226 - binary_accuracy: 1.0000
Epoch 395/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1222 - binary_accuracy: 1.0000
Epoch 396/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1219 - binary_accuracy: 1.0000
Epoch 397/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1216 - binary_accuracy: 1.0000
Epoch 398/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1212 - binary_accuracy: 1.0000
Epoch 399/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1209 - binary_accuracy: 1.0000
Epoch 400/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1206 - binary_accuracy: 1.0000
Epoch 401/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1203 - binary_accuracy: 1.0000
Epoch 402/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1199 - binary_accuracy: 1.0000
Epoch 403/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1196 - binary_accuracy: 1.0000
Epoch 404/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1193 - binary_accuracy: 1.0000
Epoch 405/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1190 - binary_accuracy: 1.0000
Epoch 406/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1186 - binary_accuracy: 1.0000
Epoch 407/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1183 - binary_accuracy: 1.0000
Epoch 408/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1180 - binary_accuracy: 1.0000
Epoch 409/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1177 - binary_accuracy: 1.0000
Epoch 410/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1173 - binary_accuracy: 1.0000
Epoch 411/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1170 - binary_accuracy: 1.0000
Epoch 412/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1167 - binary_accuracy: 1.0000
Epoch 413/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1164 - binary_accuracy: 1.0000
Epoch 414/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1160 - binary_accuracy: 1.0000
Epoch 415/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1157 - binary_accuracy: 1.0000
Epoch 416/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1154 - binary_accuracy: 1.0000
Epoch 417/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1151 - binary_accuracy: 1.0000
Epoch 418/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1148 - binary_accuracy: 1.0000
Epoch 419/1000
1/1 [==============================] - 0s 12ms/step - loss: 0.1144 - binary_accuracy: 1.0000
Epoch 420/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1141 - binary_accuracy: 1.0000
Epoch 421/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1138 - binary_accuracy: 1.0000
Epoch 422/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1135 - binary_accuracy: 1.0000
Epoch 423/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1132 - binary_accuracy: 1.0000
Epoch 424/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1129 - binary_accuracy: 1.0000
Epoch 425/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1126 - binary_accuracy: 1.0000
Epoch 426/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.1122 - binary_accuracy: 1.0000
Epoch 427/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1119 - binary_accuracy: 1.0000
Epoch 428/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1116 - binary_accuracy: 1.0000
Epoch 429/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1113 - binary_accuracy: 1.0000
Epoch 430/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1110 - binary_accuracy: 1.0000
Epoch 431/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1107 - binary_accuracy: 1.0000
Epoch 432/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1104 - binary_accuracy: 1.0000
Epoch 433/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1101 - binary_accuracy: 1.0000
Epoch 434/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1098 - binary_accuracy: 1.0000
Epoch 435/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1095 - binary_accuracy: 1.0000
Epoch 436/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1092 - binary_accuracy: 1.0000
Epoch 437/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1088 - binary_accuracy: 1.0000
Epoch 438/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1085 - binary_accuracy: 1.0000
Epoch 439/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1082 - binary_accuracy: 1.0000
Epoch 440/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1079 - binary_accuracy: 1.0000
Epoch 441/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.1076 - binary_accuracy: 1.0000
Epoch 442/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1073 - binary_accuracy: 1.0000
Epoch 443/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1070 - binary_accuracy: 1.0000
Epoch 444/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1067 - binary_accuracy: 1.0000
Epoch 445/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1064 - binary_accuracy: 1.0000
Epoch 446/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1061 - binary_accuracy: 1.0000
Epoch 447/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1058 - binary_accuracy: 1.0000
Epoch 448/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1055 - binary_accuracy: 1.0000
Epoch 449/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1052 - binary_accuracy: 1.0000
Epoch 450/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.1049 - binary_accuracy: 1.0000
Epoch 451/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1046 - binary_accuracy: 1.0000
Epoch 452/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.1043 - binary_accuracy: 1.0000
Epoch 453/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1040 - binary_accuracy: 1.0000
Epoch 454/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1037 - binary_accuracy: 1.0000
Epoch 455/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1034 - binary_accuracy: 1.0000
Epoch 456/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1031 - binary_accuracy: 1.0000
Epoch 457/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.1028 - binary_accuracy: 1.0000
Epoch 458/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1025 - binary_accuracy: 1.0000
Epoch 459/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1022 - binary_accuracy: 1.0000
Epoch 460/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1019 - binary_accuracy: 1.0000
Epoch 461/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.1016 - binary_accuracy: 1.0000
Epoch 462/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1013 - binary_accuracy: 1.0000
Epoch 463/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.1011 - binary_accuracy: 1.0000
Epoch 464/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.1008 - binary_accuracy: 1.0000
Epoch 465/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1005 - binary_accuracy: 1.0000
Epoch 466/1000
1/1 [==============================] - 0s 0s/step - loss: 0.1002 - binary_accuracy: 1.0000
Epoch 467/1000
1/1 [==============================] - 0s 16ms/step - loss: 0.0999 - binary_accuracy: 1.0000
Epoch 468/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0996 - binary_accuracy: 1.0000
Epoch 469/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0993 - binary_accuracy: 1.0000
Epoch 470/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0990 - binary_accuracy: 1.0000
Epoch 471/1000
1/1 [==============================] - 0s 16ms/step - loss: 0.0988 - binary_accuracy: 1.0000
Epoch 472/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0985 - binary_accuracy: 1.0000
Epoch 473/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0982 - binary_accuracy: 1.0000
Epoch 474/1000
1/1 [==============================] - 0s 17ms/step - loss: 0.0979 - binary_accuracy: 1.0000
Epoch 475/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0976 - binary_accuracy: 1.0000
Epoch 476/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0973 - binary_accuracy: 1.0000
Epoch 477/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0971 - binary_accuracy: 1.0000
Epoch 478/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0968 - binary_accuracy: 1.0000
Epoch 479/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0965 - binary_accuracy: 1.0000
Epoch 480/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0962 - binary_accuracy: 1.0000
Epoch 481/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0959 - binary_accuracy: 1.0000
Epoch 482/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0956 - binary_accuracy: 1.0000
Epoch 483/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0954 - binary_accuracy: 1.0000
Epoch 484/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0951 - binary_accuracy: 1.0000
Epoch 485/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0948 - binary_accuracy: 1.0000
Epoch 486/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0945 - binary_accuracy: 1.0000
Epoch 487/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0942 - binary_accuracy: 1.0000
Epoch 488/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0940 - binary_accuracy: 1.0000
Epoch 489/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0937 - binary_accuracy: 1.0000
Epoch 490/1000
1/1 [==============================] - 0s 990us/step - loss: 0.0934 - binary_accuracy: 1.0000
Epoch 491/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0931 - binary_accuracy: 1.0000
Epoch 492/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0929 - binary_accuracy: 1.0000
Epoch 493/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0926 - binary_accuracy: 1.0000
Epoch 494/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0923 - binary_accuracy: 1.0000
Epoch 495/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0921 - binary_accuracy: 1.0000
Epoch 496/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0918 - binary_accuracy: 1.0000
Epoch 497/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0915 - binary_accuracy: 1.0000
Epoch 498/1000
1/1 [==============================] - 0s 18ms/step - loss: 0.0912 - binary_accuracy: 1.0000
Epoch 499/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0910 - binary_accuracy: 1.0000
Epoch 500/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0907 - binary_accuracy: 1.0000
Epoch 501/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0904 - binary_accuracy: 1.0000
Epoch 502/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0902 - binary_accuracy: 1.0000
Epoch 503/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0899 - binary_accuracy: 1.0000
Epoch 504/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0896 - binary_accuracy: 1.0000
Epoch 505/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0894 - binary_accuracy: 1.0000
Epoch 506/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0891 - binary_accuracy: 1.0000
Epoch 507/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0889 - binary_accuracy: 1.0000
Epoch 508/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0886 - binary_accuracy: 1.0000
Epoch 509/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0883 - binary_accuracy: 1.0000
Epoch 510/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0881 - binary_accuracy: 1.0000
Epoch 511/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0878 - binary_accuracy: 1.0000
Epoch 512/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0875 - binary_accuracy: 1.0000
Epoch 513/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0873 - binary_accuracy: 1.0000
Epoch 514/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0870 - binary_accuracy: 1.0000
Epoch 515/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0868 - binary_accuracy: 1.0000
Epoch 516/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0865 - binary_accuracy: 1.0000
Epoch 517/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0862 - binary_accuracy: 1.0000
Epoch 518/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0860 - binary_accuracy: 1.0000
Epoch 519/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0857 - binary_accuracy: 1.0000
Epoch 520/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0855 - binary_accuracy: 1.0000
Epoch 521/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0852 - binary_accuracy: 1.0000
Epoch 522/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0850 - binary_accuracy: 1.0000
Epoch 523/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0847 - binary_accuracy: 1.0000
Epoch 524/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0845 - binary_accuracy: 1.0000
Epoch 525/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0842 - binary_accuracy: 1.0000
Epoch 526/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0840 - binary_accuracy: 1.0000
Epoch 527/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0837 - binary_accuracy: 1.0000
Epoch 528/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0834 - binary_accuracy: 1.0000
Epoch 529/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0832 - binary_accuracy: 1.0000
Epoch 530/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0829 - binary_accuracy: 1.0000
Epoch 531/1000
1/1 [==============================] - 0s 12ms/step - loss: 0.0827 - binary_accuracy: 1.0000
Epoch 532/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0825 - binary_accuracy: 1.0000
Epoch 533/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0822 - binary_accuracy: 1.0000
Epoch 534/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0820 - binary_accuracy: 1.0000
Epoch 535/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0817 - binary_accuracy: 1.0000
Epoch 536/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.0815 - binary_accuracy: 1.0000
Epoch 537/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0812 - binary_accuracy: 1.0000
Epoch 538/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0810 - binary_accuracy: 1.0000
Epoch 539/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.0807 - binary_accuracy: 1.0000
Epoch 540/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0805 - binary_accuracy: 1.0000
Epoch 541/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0802 - binary_accuracy: 1.0000
Epoch 542/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0800 - binary_accuracy: 1.0000
Epoch 543/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0798 - binary_accuracy: 1.0000
Epoch 544/1000
1/1 [==============================] - 0s 979us/step - loss: 0.0795 - binary_accuracy: 1.0000
Epoch 545/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0793 - binary_accuracy: 1.0000
Epoch 546/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0790 - binary_accuracy: 1.0000
Epoch 547/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0788 - binary_accuracy: 1.0000
Epoch 548/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0786 - binary_accuracy: 1.0000
Epoch 549/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0783 - binary_accuracy: 1.0000
Epoch 550/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0781 - binary_accuracy: 1.0000
Epoch 551/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0779 - binary_accuracy: 1.0000
Epoch 552/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0776 - binary_accuracy: 1.0000
Epoch 553/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0774 - binary_accuracy: 1.0000
Epoch 554/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0772 - binary_accuracy: 1.0000
Epoch 555/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0769 - binary_accuracy: 1.0000
Epoch 556/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0767 - binary_accuracy: 1.0000
Epoch 557/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0765 - binary_accuracy: 1.0000
Epoch 558/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0762 - binary_accuracy: 1.0000
Epoch 559/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0760 - binary_accuracy: 1.0000
Epoch 560/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0758 - binary_accuracy: 1.0000
Epoch 561/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0756 - binary_accuracy: 1.0000
Epoch 562/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0753 - binary_accuracy: 1.0000
Epoch 563/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.0751 - binary_accuracy: 1.0000
Epoch 564/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0749 - binary_accuracy: 1.0000
Epoch 565/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0746 - binary_accuracy: 1.0000
Epoch 566/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0744 - binary_accuracy: 1.0000
Epoch 567/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0742 - binary_accuracy: 1.0000
Epoch 568/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0740 - binary_accuracy: 1.0000
Epoch 569/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0737 - binary_accuracy: 1.0000
Epoch 570/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0735 - binary_accuracy: 1.0000
Epoch 571/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0733 - binary_accuracy: 1.0000
Epoch 572/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0731 - binary_accuracy: 1.0000
Epoch 573/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0728 - binary_accuracy: 1.0000
Epoch 574/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0726 - binary_accuracy: 1.0000
Epoch 575/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0724 - binary_accuracy: 1.0000
Epoch 576/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0722 - binary_accuracy: 1.0000
Epoch 577/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0720 - binary_accuracy: 1.0000
Epoch 578/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0718 - binary_accuracy: 1.0000
Epoch 579/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0715 - binary_accuracy: 1.0000
Epoch 580/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0713 - binary_accuracy: 1.0000
Epoch 581/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0711 - binary_accuracy: 1.0000
Epoch 582/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0709 - binary_accuracy: 1.0000
Epoch 583/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0707 - binary_accuracy: 1.0000
Epoch 584/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0704 - binary_accuracy: 1.0000
Epoch 585/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0702 - binary_accuracy: 1.0000
Epoch 586/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0700 - binary_accuracy: 1.0000
Epoch 587/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0698 - binary_accuracy: 1.0000
Epoch 588/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0696 - binary_accuracy: 1.0000
Epoch 589/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0694 - binary_accuracy: 1.0000
Epoch 590/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0692 - binary_accuracy: 1.0000
Epoch 591/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0690 - binary_accuracy: 1.0000
Epoch 592/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0687 - binary_accuracy: 1.0000
Epoch 593/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0685 - binary_accuracy: 1.0000
Epoch 594/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0683 - binary_accuracy: 1.0000
Epoch 595/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0681 - binary_accuracy: 1.0000
Epoch 596/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0679 - binary_accuracy: 1.0000
Epoch 597/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0677 - binary_accuracy: 1.0000
Epoch 598/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0675 - binary_accuracy: 1.0000
Epoch 599/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0673 - binary_accuracy: 1.0000
Epoch 600/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0671 - binary_accuracy: 1.0000
Epoch 601/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0669 - binary_accuracy: 1.0000
Epoch 602/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0667 - binary_accuracy: 1.0000
Epoch 603/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0665 - binary_accuracy: 1.0000
Epoch 604/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0663 - binary_accuracy: 1.0000
Epoch 605/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0661 - binary_accuracy: 1.0000
Epoch 606/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0659 - binary_accuracy: 1.0000
Epoch 607/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0657 - binary_accuracy: 1.0000
Epoch 608/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0655 - binary_accuracy: 1.0000
Epoch 609/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0653 - binary_accuracy: 1.0000
Epoch 610/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0651 - binary_accuracy: 1.0000
Epoch 611/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0649 - binary_accuracy: 1.0000
Epoch 612/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0647 - binary_accuracy: 1.0000
Epoch 613/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0645 - binary_accuracy: 1.0000
Epoch 614/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0643 - binary_accuracy: 1.0000
Epoch 615/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0641 - binary_accuracy: 1.0000
Epoch 616/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0639 - binary_accuracy: 1.0000
Epoch 617/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0637 - binary_accuracy: 1.0000
Epoch 618/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0635 - binary_accuracy: 1.0000
Epoch 619/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0633 - binary_accuracy: 1.0000
Epoch 620/1000
1/1 [==============================] - 0s 335us/step - loss: 0.0631 - binary_accuracy: 1.0000
Epoch 621/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0629 - binary_accuracy: 1.0000
Epoch 622/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0627 - binary_accuracy: 1.0000
Epoch 623/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0625 - binary_accuracy: 1.0000
Epoch 624/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0624 - binary_accuracy: 1.0000
Epoch 625/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0622 - binary_accuracy: 1.0000
Epoch 626/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0620 - binary_accuracy: 1.0000
Epoch 627/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0618 - binary_accuracy: 1.0000
Epoch 628/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0616 - binary_accuracy: 1.0000
Epoch 629/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0614 - binary_accuracy: 1.0000
Epoch 630/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0612 - binary_accuracy: 1.0000
Epoch 631/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0610 - binary_accuracy: 1.0000
Epoch 632/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0608 - binary_accuracy: 1.0000
Epoch 633/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0607 - binary_accuracy: 1.0000
Epoch 634/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0605 - binary_accuracy: 1.0000
Epoch 635/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0603 - binary_accuracy: 1.0000
Epoch 636/1000
1/1 [==============================] - 0s 12ms/step - loss: 0.0601 - binary_accuracy: 1.0000
Epoch 637/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0599 - binary_accuracy: 1.0000
Epoch 638/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0597 - binary_accuracy: 1.0000
Epoch 639/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0596 - binary_accuracy: 1.0000
Epoch 640/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0594 - binary_accuracy: 1.0000
Epoch 641/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0592 - binary_accuracy: 1.0000
Epoch 642/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0590 - binary_accuracy: 1.0000
Epoch 643/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0588 - binary_accuracy: 1.0000
Epoch 644/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0587 - binary_accuracy: 1.0000
Epoch 645/1000
1/1 [==============================] - 0s 17ms/step - loss: 0.0585 - binary_accuracy: 1.0000
Epoch 646/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0583 - binary_accuracy: 1.0000
Epoch 647/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0581 - binary_accuracy: 1.0000
Epoch 648/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0580 - binary_accuracy: 1.0000
Epoch 649/1000
1/1 [==============================] - 0s 12ms/step - loss: 0.0578 - binary_accuracy: 1.0000
Epoch 650/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0576 - binary_accuracy: 1.0000
Epoch 651/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0574 - binary_accuracy: 1.0000
Epoch 652/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0572 - binary_accuracy: 1.0000
Epoch 653/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0571 - binary_accuracy: 1.0000
Epoch 654/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0569 - binary_accuracy: 1.0000
Epoch 655/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0567 - binary_accuracy: 1.0000
Epoch 656/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0566 - binary_accuracy: 1.0000
Epoch 657/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0564 - binary_accuracy: 1.0000
Epoch 658/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0562 - binary_accuracy: 1.0000
Epoch 659/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0560 - binary_accuracy: 1.0000
Epoch 660/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0559 - binary_accuracy: 1.0000
Epoch 661/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0557 - binary_accuracy: 1.0000
Epoch 662/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0555 - binary_accuracy: 1.0000
Epoch 663/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0554 - binary_accuracy: 1.0000
Epoch 664/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0552 - binary_accuracy: 1.0000
Epoch 665/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0550 - binary_accuracy: 1.0000
Epoch 666/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0549 - binary_accuracy: 1.0000
Epoch 667/1000
1/1 [==============================] - 0s 14ms/step - loss: 0.0547 - binary_accuracy: 1.0000
Epoch 668/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0545 - binary_accuracy: 1.0000
Epoch 669/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0544 - binary_accuracy: 1.0000
Epoch 670/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0542 - binary_accuracy: 1.0000
Epoch 671/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0540 - binary_accuracy: 1.0000
Epoch 672/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0539 - binary_accuracy: 1.0000
Epoch 673/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0537 - binary_accuracy: 1.0000
Epoch 674/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0535 - binary_accuracy: 1.0000
Epoch 675/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0534 - binary_accuracy: 1.0000
Epoch 676/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0532 - binary_accuracy: 1.0000
Epoch 677/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0531 - binary_accuracy: 1.0000
Epoch 678/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0529 - binary_accuracy: 1.0000
Epoch 679/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0527 - binary_accuracy: 1.0000
Epoch 680/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0526 - binary_accuracy: 1.0000
Epoch 681/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0524 - binary_accuracy: 1.0000
Epoch 682/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0523 - binary_accuracy: 1.0000
Epoch 683/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0521 - binary_accuracy: 1.0000
Epoch 684/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0519 - binary_accuracy: 1.0000
Epoch 685/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0518 - binary_accuracy: 1.0000
Epoch 686/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0516 - binary_accuracy: 1.0000
Epoch 687/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0515 - binary_accuracy: 1.0000
Epoch 688/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0513 - binary_accuracy: 1.0000
Epoch 689/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0512 - binary_accuracy: 1.0000
Epoch 690/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0510 - binary_accuracy: 1.0000
Epoch 691/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0508 - binary_accuracy: 1.0000
Epoch 692/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.0507 - binary_accuracy: 1.0000
Epoch 693/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0505 - binary_accuracy: 1.0000
Epoch 694/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0504 - binary_accuracy: 1.0000
Epoch 695/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0502 - binary_accuracy: 1.0000
Epoch 696/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0501 - binary_accuracy: 1.0000
Epoch 697/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0499 - binary_accuracy: 1.0000
Epoch 698/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0498 - binary_accuracy: 1.0000
Epoch 699/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0496 - binary_accuracy: 1.0000
Epoch 700/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0495 - binary_accuracy: 1.0000
Epoch 701/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0493 - binary_accuracy: 1.0000
Epoch 702/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0492 - binary_accuracy: 1.0000
Epoch 703/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0490 - binary_accuracy: 1.0000
Epoch 704/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0489 - binary_accuracy: 1.0000
Epoch 705/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0487 - binary_accuracy: 1.0000
Epoch 706/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0486 - binary_accuracy: 1.0000
Epoch 707/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0484 - binary_accuracy: 1.0000
Epoch 708/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0483 - binary_accuracy: 1.0000
Epoch 709/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0481 - binary_accuracy: 1.0000
Epoch 710/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0480 - binary_accuracy: 1.0000
Epoch 711/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0479 - binary_accuracy: 1.0000
Epoch 712/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0477 - binary_accuracy: 1.0000
Epoch 713/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0476 - binary_accuracy: 1.0000
Epoch 714/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0474 - binary_accuracy: 1.0000
Epoch 715/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0473 - binary_accuracy: 1.0000
Epoch 716/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0471 - binary_accuracy: 1.0000
Epoch 717/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0470 - binary_accuracy: 1.0000
Epoch 718/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0469 - binary_accuracy: 1.0000
Epoch 719/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0467 - binary_accuracy: 1.0000
Epoch 720/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0466 - binary_accuracy: 1.0000
Epoch 721/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0464 - binary_accuracy: 1.0000
Epoch 722/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0463 - binary_accuracy: 1.0000
Epoch 723/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0462 - binary_accuracy: 1.0000
Epoch 724/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0460 - binary_accuracy: 1.0000
Epoch 725/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0459 - binary_accuracy: 1.0000
Epoch 726/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0457 - binary_accuracy: 1.0000
Epoch 727/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0456 - binary_accuracy: 1.0000
Epoch 728/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0455 - binary_accuracy: 1.0000
Epoch 729/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0453 - binary_accuracy: 1.0000
Epoch 730/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0452 - binary_accuracy: 1.0000
Epoch 731/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0451 - binary_accuracy: 1.0000
Epoch 732/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0449 - binary_accuracy: 1.0000
Epoch 733/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0448 - binary_accuracy: 1.0000
Epoch 734/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0447 - binary_accuracy: 1.0000
Epoch 735/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0445 - binary_accuracy: 1.0000
Epoch 736/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0444 - binary_accuracy: 1.0000
Epoch 737/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0443 - binary_accuracy: 1.0000
Epoch 738/1000
1/1 [==============================] - 0s 500us/step - loss: 0.0441 - binary_accuracy: 1.0000
Epoch 739/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0440 - binary_accuracy: 1.0000
Epoch 740/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0439 - binary_accuracy: 1.0000
Epoch 741/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0437 - binary_accuracy: 1.0000
Epoch 742/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0436 - binary_accuracy: 1.0000
Epoch 743/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0435 - binary_accuracy: 1.0000
Epoch 744/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0434 - binary_accuracy: 1.0000
Epoch 745/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0432 - binary_accuracy: 1.0000
Epoch 746/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0431 - binary_accuracy: 1.0000
Epoch 747/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0430 - binary_accuracy: 1.0000
Epoch 748/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0428 - binary_accuracy: 1.0000
Epoch 749/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0427 - binary_accuracy: 1.0000
Epoch 750/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0426 - binary_accuracy: 1.0000
Epoch 751/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0425 - binary_accuracy: 1.0000
Epoch 752/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0423 - binary_accuracy: 1.0000
Epoch 753/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0422 - binary_accuracy: 1.0000
Epoch 754/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0421 - binary_accuracy: 1.0000
Epoch 755/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0419 - binary_accuracy: 1.0000
Epoch 756/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0418 - binary_accuracy: 1.0000
Epoch 757/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0417 - binary_accuracy: 1.0000
Epoch 758/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0416 - binary_accuracy: 1.0000
Epoch 759/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0415 - binary_accuracy: 1.0000
Epoch 760/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0413 - binary_accuracy: 1.0000
Epoch 761/1000
1/1 [==============================] - 0s 847us/step - loss: 0.0412 - binary_accuracy: 1.0000
Epoch 762/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0411 - binary_accuracy: 1.0000
Epoch 763/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0410 - binary_accuracy: 1.0000
Epoch 764/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0408 - binary_accuracy: 1.0000
Epoch 765/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0407 - binary_accuracy: 1.0000
Epoch 766/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0406 - binary_accuracy: 1.0000
Epoch 767/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0405 - binary_accuracy: 1.0000
Epoch 768/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0404 - binary_accuracy: 1.0000
Epoch 769/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0402 - binary_accuracy: 1.0000
Epoch 770/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0401 - binary_accuracy: 1.0000
Epoch 771/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0400 - binary_accuracy: 1.0000
Epoch 772/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0399 - binary_accuracy: 1.0000
Epoch 773/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0398 - binary_accuracy: 1.0000
Epoch 774/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0397 - binary_accuracy: 1.0000
Epoch 775/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0395 - binary_accuracy: 1.0000
Epoch 776/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0394 - binary_accuracy: 1.0000
Epoch 777/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0393 - binary_accuracy: 1.0000
Epoch 778/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0392 - binary_accuracy: 1.0000
Epoch 779/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0391 - binary_accuracy: 1.0000
Epoch 780/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0390 - binary_accuracy: 1.0000
Epoch 781/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0389 - binary_accuracy: 1.0000
Epoch 782/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0387 - binary_accuracy: 1.0000
Epoch 783/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0386 - binary_accuracy: 1.0000
Epoch 784/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0385 - binary_accuracy: 1.0000
Epoch 785/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.0384 - binary_accuracy: 1.0000
Epoch 786/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0383 - binary_accuracy: 1.0000
Epoch 787/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0382 - binary_accuracy: 1.0000
Epoch 788/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0381 - binary_accuracy: 1.0000
Epoch 789/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0380 - binary_accuracy: 1.0000
Epoch 790/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0378 - binary_accuracy: 1.0000
Epoch 791/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0377 - binary_accuracy: 1.0000
Epoch 792/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0376 - binary_accuracy: 1.0000
Epoch 793/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0375 - binary_accuracy: 1.0000
Epoch 794/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0374 - binary_accuracy: 1.0000
Epoch 795/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0373 - binary_accuracy: 1.0000
Epoch 796/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0372 - binary_accuracy: 1.0000
Epoch 797/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0371 - binary_accuracy: 1.0000
Epoch 798/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0370 - binary_accuracy: 1.0000
Epoch 799/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0369 - binary_accuracy: 1.0000
Epoch 800/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0367 - binary_accuracy: 1.0000
Epoch 801/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0366 - binary_accuracy: 1.0000
Epoch 802/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0365 - binary_accuracy: 1.0000
Epoch 803/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0364 - binary_accuracy: 1.0000
Epoch 804/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0363 - binary_accuracy: 1.0000
Epoch 805/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0362 - binary_accuracy: 1.0000
Epoch 806/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0361 - binary_accuracy: 1.0000
Epoch 807/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0360 - binary_accuracy: 1.0000
Epoch 808/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0359 - binary_accuracy: 1.0000
Epoch 809/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0358 - binary_accuracy: 1.0000
Epoch 810/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0357 - binary_accuracy: 1.0000
Epoch 811/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0356 - binary_accuracy: 1.0000
Epoch 812/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0355 - binary_accuracy: 1.0000
Epoch 813/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0354 - binary_accuracy: 1.0000
Epoch 814/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0353 - binary_accuracy: 1.0000
Epoch 815/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0352 - binary_accuracy: 1.0000
Epoch 816/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0351 - binary_accuracy: 1.0000
Epoch 817/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0350 - binary_accuracy: 1.0000
Epoch 818/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0349 - binary_accuracy: 1.0000
Epoch 819/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0348 - binary_accuracy: 1.0000
Epoch 820/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0347 - binary_accuracy: 1.0000
Epoch 821/1000
1/1 [==============================] - 0s 13ms/step - loss: 0.0346 - binary_accuracy: 1.0000
Epoch 822/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0345 - binary_accuracy: 1.0000
Epoch 823/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0344 - binary_accuracy: 1.0000
Epoch 824/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0343 - binary_accuracy: 1.0000
Epoch 825/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0342 - binary_accuracy: 1.0000
Epoch 826/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0341 - binary_accuracy: 1.0000
Epoch 827/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0340 - binary_accuracy: 1.0000
Epoch 828/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0339 - binary_accuracy: 1.0000
Epoch 829/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0338 - binary_accuracy: 1.0000
Epoch 830/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0337 - binary_accuracy: 1.0000
Epoch 831/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0336 - binary_accuracy: 1.0000
Epoch 832/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0335 - binary_accuracy: 1.0000
Epoch 833/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0334 - binary_accuracy: 1.0000
Epoch 834/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0333 - binary_accuracy: 1.0000
Epoch 835/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0332 - binary_accuracy: 1.0000
Epoch 836/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0331 - binary_accuracy: 1.0000
Epoch 837/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0330 - binary_accuracy: 1.0000
Epoch 838/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0329 - binary_accuracy: 1.0000
Epoch 839/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0328 - binary_accuracy: 1.0000
Epoch 840/1000
1/1 [==============================] - 0s 7ms/step - loss: 0.0328 - binary_accuracy: 1.0000
Epoch 841/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0327 - binary_accuracy: 1.0000
Epoch 842/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0326 - binary_accuracy: 1.0000
Epoch 843/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0325 - binary_accuracy: 1.0000
Epoch 844/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0324 - binary_accuracy: 1.0000
Epoch 845/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0323 - binary_accuracy: 1.0000
Epoch 846/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0322 - binary_accuracy: 1.0000
Epoch 847/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0321 - binary_accuracy: 1.0000
Epoch 848/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0320 - binary_accuracy: 1.0000
Epoch 849/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0319 - binary_accuracy: 1.0000
Epoch 850/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0318 - binary_accuracy: 1.0000
Epoch 851/1000
1/1 [==============================] - 0s 659us/step - loss: 0.0317 - binary_accuracy: 1.0000
Epoch 852/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0317 - binary_accuracy: 1.0000
Epoch 853/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0316 - binary_accuracy: 1.0000
Epoch 854/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0315 - binary_accuracy: 1.0000
Epoch 855/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0314 - binary_accuracy: 1.0000
Epoch 856/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0313 - binary_accuracy: 1.0000
Epoch 857/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0312 - binary_accuracy: 1.0000
Epoch 858/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0311 - binary_accuracy: 1.0000
Epoch 859/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0310 - binary_accuracy: 1.0000
Epoch 860/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.0309 - binary_accuracy: 1.0000
Epoch 861/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0309 - binary_accuracy: 1.0000
Epoch 862/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0308 - binary_accuracy: 1.0000
Epoch 863/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0307 - binary_accuracy: 1.0000
Epoch 864/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0306 - binary_accuracy: 1.0000
Epoch 865/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0305 - binary_accuracy: 1.0000
Epoch 866/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0304 - binary_accuracy: 1.0000
Epoch 867/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0303 - binary_accuracy: 1.0000
Epoch 868/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0303 - binary_accuracy: 1.0000
Epoch 869/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0302 - binary_accuracy: 1.0000
Epoch 870/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0301 - binary_accuracy: 1.0000
Epoch 871/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0300 - binary_accuracy: 1.0000
Epoch 872/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0299 - binary_accuracy: 1.0000
Epoch 873/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0298 - binary_accuracy: 1.0000
Epoch 874/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0297 - binary_accuracy: 1.0000
Epoch 875/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0297 - binary_accuracy: 1.0000
Epoch 876/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0296 - binary_accuracy: 1.0000
Epoch 877/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0295 - binary_accuracy: 1.0000
Epoch 878/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0294 - binary_accuracy: 1.0000
Epoch 879/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0293 - binary_accuracy: 1.0000
Epoch 880/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0293 - binary_accuracy: 1.0000
Epoch 881/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0292 - binary_accuracy: 1.0000
Epoch 882/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0291 - binary_accuracy: 1.0000
Epoch 883/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0290 - binary_accuracy: 1.0000
Epoch 884/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0289 - binary_accuracy: 1.0000
Epoch 885/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0289 - binary_accuracy: 1.0000
Epoch 886/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0288 - binary_accuracy: 1.0000
Epoch 887/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0287 - binary_accuracy: 1.0000
Epoch 888/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0286 - binary_accuracy: 1.0000
Epoch 889/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0285 - binary_accuracy: 1.0000
Epoch 890/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0285 - binary_accuracy: 1.0000
Epoch 891/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0284 - binary_accuracy: 1.0000
Epoch 892/1000
1/1 [==============================] - 0s 317us/step - loss: 0.0283 - binary_accuracy: 1.0000
Epoch 893/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0282 - binary_accuracy: 1.0000
Epoch 894/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0281 - binary_accuracy: 1.0000
Epoch 895/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0281 - binary_accuracy: 1.0000
Epoch 896/1000
1/1 [==============================] - 0s 13ms/step - loss: 0.0280 - binary_accuracy: 1.0000
Epoch 897/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0279 - binary_accuracy: 1.0000
Epoch 898/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0278 - binary_accuracy: 1.0000
Epoch 899/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0278 - binary_accuracy: 1.0000
Epoch 900/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0277 - binary_accuracy: 1.0000
Epoch 901/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0276 - binary_accuracy: 1.0000
Epoch 902/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0275 - binary_accuracy: 1.0000
Epoch 903/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0275 - binary_accuracy: 1.0000
Epoch 904/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0274 - binary_accuracy: 1.0000
Epoch 905/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0273 - binary_accuracy: 1.0000
Epoch 906/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0272 - binary_accuracy: 1.0000
Epoch 907/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0272 - binary_accuracy: 1.0000
Epoch 908/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0271 - binary_accuracy: 1.0000
Epoch 909/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0270 - binary_accuracy: 1.0000
Epoch 910/1000
1/1 [==============================] - 0s 12ms/step - loss: 0.0269 - binary_accuracy: 1.0000
Epoch 911/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0269 - binary_accuracy: 1.0000
Epoch 912/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0268 - binary_accuracy: 1.0000
Epoch 913/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0267 - binary_accuracy: 1.0000
Epoch 914/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0266 - binary_accuracy: 1.0000
Epoch 915/1000
1/1 [==============================] - 0s 502us/step - loss: 0.0266 - binary_accuracy: 1.0000
Epoch 916/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0265 - binary_accuracy: 1.0000
Epoch 917/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0264 - binary_accuracy: 1.0000
Epoch 918/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0263 - binary_accuracy: 1.0000
Epoch 919/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0263 - binary_accuracy: 1.0000
Epoch 920/1000
1/1 [==============================] - 0s 11ms/step - loss: 0.0262 - binary_accuracy: 1.0000
Epoch 921/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0261 - binary_accuracy: 1.0000
Epoch 922/1000
1/1 [==============================] - 0s 890us/step - loss: 0.0261 - binary_accuracy: 1.0000
Epoch 923/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0260 - binary_accuracy: 1.0000
Epoch 924/1000
1/1 [==============================] - 0s 8ms/step - loss: 0.0259 - binary_accuracy: 1.0000
Epoch 925/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0259 - binary_accuracy: 1.0000
Epoch 926/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0258 - binary_accuracy: 1.0000
Epoch 927/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0257 - binary_accuracy: 1.0000
Epoch 928/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0256 - binary_accuracy: 1.0000
Epoch 929/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0256 - binary_accuracy: 1.0000
Epoch 930/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0255 - binary_accuracy: 1.0000
Epoch 931/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0254 - binary_accuracy: 1.0000
Epoch 932/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0254 - binary_accuracy: 1.0000
Epoch 933/1000
1/1 [==============================] - 0s 498us/step - loss: 0.0253 - binary_accuracy: 1.0000
Epoch 934/1000
1/1 [==============================] - 0s 499us/step - loss: 0.0252 - binary_accuracy: 1.0000
Epoch 935/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0252 - binary_accuracy: 1.0000
Epoch 936/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0251 - binary_accuracy: 1.0000
Epoch 937/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0250 - binary_accuracy: 1.0000
Epoch 938/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0250 - binary_accuracy: 1.0000
Epoch 939/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0249 - binary_accuracy: 1.0000
Epoch 940/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0248 - binary_accuracy: 1.0000
Epoch 941/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0248 - binary_accuracy: 1.0000
Epoch 942/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0247 - binary_accuracy: 1.0000
Epoch 943/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0246 - binary_accuracy: 1.0000
Epoch 944/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0246 - binary_accuracy: 1.0000
Epoch 945/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0245 - binary_accuracy: 1.0000
Epoch 946/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0244 - binary_accuracy: 1.0000
Epoch 947/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0244 - binary_accuracy: 1.0000
Epoch 948/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0243 - binary_accuracy: 1.0000
Epoch 949/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0242 - binary_accuracy: 1.0000
Epoch 950/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0242 - binary_accuracy: 1.0000
Epoch 951/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0241 - binary_accuracy: 1.0000
Epoch 952/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0240 - binary_accuracy: 1.0000
Epoch 953/1000
1/1 [==============================] - 0s 732us/step - loss: 0.0240 - binary_accuracy: 1.0000
Epoch 954/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0239 - binary_accuracy: 1.0000
Epoch 955/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0239 - binary_accuracy: 1.0000
Epoch 956/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0238 - binary_accuracy: 1.0000
Epoch 957/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0237 - binary_accuracy: 1.0000
Epoch 958/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0237 - binary_accuracy: 1.0000
Epoch 959/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0236 - binary_accuracy: 1.0000
Epoch 960/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0235 - binary_accuracy: 1.0000
Epoch 961/1000
1/1 [==============================] - 0s 9ms/step - loss: 0.0235 - binary_accuracy: 1.0000
Epoch 962/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0234 - binary_accuracy: 1.0000
Epoch 963/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0234 - binary_accuracy: 1.0000
Epoch 964/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0233 - binary_accuracy: 1.0000
Epoch 965/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0232 - binary_accuracy: 1.0000
Epoch 966/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0232 - binary_accuracy: 1.0000
Epoch 967/1000
1/1 [==============================] - 0s 21ms/step - loss: 0.0231 - binary_accuracy: 1.0000
Epoch 968/1000
1/1 [==============================] - 0s 10ms/step - loss: 0.0230 - binary_accuracy: 1.0000
Epoch 969/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0230 - binary_accuracy: 1.0000
Epoch 970/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0229 - binary_accuracy: 1.0000
Epoch 971/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0229 - binary_accuracy: 1.0000
Epoch 972/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0228 - binary_accuracy: 1.0000
Epoch 973/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0228 - binary_accuracy: 1.0000
Epoch 974/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0227 - binary_accuracy: 1.0000
Epoch 975/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0226 - binary_accuracy: 1.0000
Epoch 976/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0226 - binary_accuracy: 1.0000
Epoch 977/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0225 - binary_accuracy: 1.0000
Epoch 978/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0225 - binary_accuracy: 1.0000
Epoch 979/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0224 - binary_accuracy: 1.0000
Epoch 980/1000
1/1 [==============================] - 0s 3ms/step - loss: 0.0223 - binary_accuracy: 1.0000
Epoch 981/1000
1/1 [==============================] - 0s 4ms/step - loss: 0.0223 - binary_accuracy: 1.0000
Epoch 982/1000
1/1 [==============================] - 0s 1ms/step - loss: 0.0222 - binary_accuracy: 1.0000
Epoch 983/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0222 - binary_accuracy: 1.0000
Epoch 984/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0221 - binary_accuracy: 1.0000
Epoch 985/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0220 - binary_accuracy: 1.0000
Epoch 986/1000
1/1 [==============================] - 0s 15ms/step - loss: 0.0220 - binary_accuracy: 1.0000
Epoch 987/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0219 - binary_accuracy: 1.0000
Epoch 988/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0219 - binary_accuracy: 1.0000
Epoch 989/1000
1/1 [==============================] - 0s 6ms/step - loss: 0.0218 - binary_accuracy: 1.0000
Epoch 990/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0218 - binary_accuracy: 1.0000
Epoch 991/1000
1/1 [==============================] - 0s 2ms/step - loss: 0.0217 - binary_accuracy: 1.0000
Epoch 992/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0216 - binary_accuracy: 1.0000
Epoch 993/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0216 - binary_accuracy: 1.0000
Epoch 994/1000
1/1 [==============================] - 0s 864us/step - loss: 0.0215 - binary_accuracy: 1.0000
Epoch 995/1000
1/1 [==============================] - 0s 5ms/step - loss: 0.0215 - binary_accuracy: 1.0000
Epoch 996/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0214 - binary_accuracy: 1.0000
Epoch 997/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0214 - binary_accuracy: 1.0000
Epoch 998/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0213 - binary_accuracy: 1.0000
Epoch 999/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0213 - binary_accuracy: 1.0000
Epoch 1000/1000
1/1 [==============================] - 0s 0s/step - loss: 0.0212 - binary_accuracy: 1.0000
1/1 [==============================] - 0s 57ms/step - loss: 0.0212 - binary_accuracy: 1.0000

binary_accuracy: 100.00%
1/1 [==============================] - 0s 32ms/step
[[0.]
 [1.]
 [1.]
 [0.]]

```


## Result
Thus the python program successully implemented XOR logic gate.
