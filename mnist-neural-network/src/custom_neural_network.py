import numpy as np
import pandas as pd #pandas to read datasheet

# from matplotlib import pyplot as plt 

data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')


print(data.head())

#preprocessing the data:-
data = np.array(data)

np.random.shuffle(data)

x = data[:, 1:]     #features
y = data[:, 0]      #labels
x = x/255.0         #normalization

print("x shape = ", x.shape)
print("y shape = ", y.shape)

#for testing & training  :-
x_train = x[1000:]
y_train = y[1000:]
x_test = x[:1000]
y_test = y[:1000]

print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

class Activation_softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims = True)
		self.output = probabilities

class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		print("data_loss = ", data_loss)
		return data_loss

class Loss_CategoricalCrossentropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)

		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]
			print("correct confidences = ", correct_confidences)

		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
			print("correct confidences = ", correct_confidences)
		negative_log_likelihoods = -np.log(correct_confidences)
		print("negative log likelihood = ", negative_log_likelihoods)
		return negative_log_likelihoods
	
layer1 = Layer_Dense(n_inputs = 784, n_neurons = 128)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(n_inputs = 128, n_neurons = 64)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(n_inputs = 64, n_neurons = 10)
activation3 = Activation_softmax()

num_epoch = 150 #increase or decrease the epoch for changing the accuracy of the model.
# learning_rate = 0.01
loss_function = Loss_CategoricalCrossentropy()


initial_lr = 0.1
decay = initial_lr / num_epoch

for epoch in range(num_epoch):
	learning_rate = initial_lr * (1 / (1 + decay * epoch)) #decays learning rate with per epochs.

	layer1.forward(x_train)
	activation1.forward(layer1.output)

	layer2.forward(activation1.output)
	activation2.forward(layer2.output)

	layer3.forward(activation2.output)
	activation3.forward(layer3.output)
	predictions = activation3.output

	#ensuring that the y_train data is in one hot coded.
	y_train_one_hot = np.zeros((y_train.size, y_train.max() + 1))
	y_train_one_hot[np.arange(y_train.size), y_train] = 1

	loss = loss_function.calculate(activation3.output, y_train_one_hot)

	# print("Output probabilities :\n", activation3.output)

	predicted_labels = np.argmax(predictions, axis=1)
	accuracy = np.mean(predicted_labels == y_train)

	#implementing backward propagation and finding the gradient of loss w.r.t the output of the last layer.

	dZ3 = predictions - y_train_one_hot #derivative of the loss w.r.t. Z3

	#gradients for output layer i.e layer 3.
	#{original owner: KunwarPrabhat}
	dw3 = np.dot(activation2.output.T, dZ3) / x_train.shape[0] #derivative w.r.t w3
	db3 = np.sum(dZ3, axis=0, keepdims=True) / x_train.shape[0] #derivative w.r.t b3

	#gradients w.r.t activation 2 (output of layer 2)
	dA2 = np.dot(dZ3, layer3.weights.T)

	#gradient of layer 2
	dZ2 = dA2 * (activation2.output > 0 )
	dw2 = np.dot(activation1.output.T, dZ2) / x_train.shape[0]
	db2 = np.sum(dZ2, axis=0, keepdims=True) / x_train.shape[0]

	#gradients w.r.t activation 1 (output of layer 1)
	dA1 = np.dot(dZ2, layer2.weights.T)

	#gradients for layer 1 
	dZ1 = dA1 * (activation1.output > 0)
	dw1 = np.dot(x_train.T, dZ1) / x_train.shape[0]
	db1 = np.sum(dZ1, axis=0, keepdims=True) / x_train.shape[0]



	#updating weights 
	layer3.weights -= learning_rate * dw3
	layer3.biases -= learning_rate * db3

	layer2.weights -= learning_rate * dw2
	layer2.biases -= learning_rate * db2

	layer1.weights -= learning_rate * dw1
	layer1.biases -= learning_rate * db1

	print("Loss", loss)
	print("accuracy - ", accuracy*100)
	print("Epoch : ", epoch)

#uncomment to Save weights and biases of each layer to files in 'Trained_Weights' folder

# np.save('Trained_Weights/layer1_weights.npy', layer1.weights)
# np.save('Trained_Weights/layer1_biases.npy', layer1.biases)

# np.save('Trained_Weights/layer2_weights.npy', layer2.weights)
# np.save('Trained_Weights/layer2_biases.npy', layer2.biases)

# np.save('Trained_Weights/layer3_weights.npy', layer3.weights)
# np.save('Trained_Weights/layer3_biases.npy', layer3.biases)

# print("Model weights and biases saved successfully.")


#testing with test dataset :-

layer1.forward(x_test)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3.forward(activation2.output)
activation3.forward(layer3.output)

predictions = np.argmax(activation3.output, axis=1)
accuracy = np.mean(predictions == y_test)


print(f"Test Accuracy: {accuracy * 100}%\n")
print("Git - KunwarPrabhat")

#manual testing begins :-
print("Testing on first 1000 isolated elements from train.csv files :- \n")

selected_indices = [0, 5, 8, 10, 12, 16, 20, 25, 28, 30, 34, 36, 40, 41, 48, 50, 52, 55, 58, 60, 61, 62, 63, ] #edit this field to random indices to test.
selected_samples = x_test[selected_indices]
selected_labels = y_test[selected_indices]
predictions = []

for i, sample in enumerate(selected_samples):
    sample = sample.reshape(1, -1) #reshaping the array if it's 2D to a linear array.

	#moving forward with trained weights and biases.
    layer1.forward(sample)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    predicted_label = np.argmax(activation3.output, axis=1)[0]
    actual_label = selected_labels[i]

    predictions.append(predicted_label)

    print(f"Sample {i+1}:")
    print(f"  Actual Label: {actual_label}")
    print(f"  Predicted Label: {predicted_label}\n")

predictions = np.array(predictions)
accuracy = np.mean(predictions == selected_labels) * 100
print(f" Overall Accuracy : {accuracy:.2f}%\n")

#test on test.csv file (blind prediction):-

# print("Performing test on test.csv file (blind predictions): - \n")

# testing_data = np.array(testing_data)

# x1 = testing_data

# x1 = x1/255.0 

# for i, sample in enumerate(x1):
# 	sample = sample.reshape(1, -1)

# 	layer1.forward(sample)
# 	activation1.forward(layer1.output)

# 	layer2.forward(activation1.output)
# 	activation2.forward(layer2.output)

# 	layer3.forward(activation2.output)
# 	activation3.forward(layer3.output)

# 	predicted_label = np.argmax(activation3.output, axis=1)[0]

# 	print(f"sample {i+1}:")
# 	print(f" Predcted label : {predicted_label}\n")



