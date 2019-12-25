import numpy as np
import scipy.io.wavfile as wav
import os
import speechpy
import sys
from sklearn.model_selection import train_test_split
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten,BatchNormalization, Activation, MaxPooling2D
class_labels = ["Neutral", "Angry", "Happy", "Sad"]
nClasses=len(class_labels)
mslen = 32000
from keras.utils import to_categorical
def read_wav(filename):
    """
    Read the wav file and return corresponding data
    :param filename: name of the file
    :return: return tuple containing sampling frequency and signal
    """
    return wav.read(filename)


def get_data(dataset_path, flatten=True, mfcc_len=39):
    """
    Read the files get the data perform the test-train split and return them to the caller
    :param dataset_path: path to the dataset folder
    :param mfcc_len: Number of mfcc features to take for each frame
    :param flatten: Boolean specifying whether to flatten the data or not
    :return: 4 arrays, x_train x_test y_train y_test
    """
    data = []
    labels = []
    max_fs = 0
    s = 0
    cnt = 0
    cur_dir = os.getcwd()
    print('curdir', cur_dir)
    os.chdir(dataset_path)
    for i, directory in enumerate(class_labels):
        print ("started reading folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            max_fs = max(max_fs, fs)
            s_len = len(signal)
            # pad the signals to have same size if lesser than required
            # else slice them
            if s_len < mslen:
                pad_len = mslen - s_len
                pad_rem = pad_len % 2
                pad_len /= 2
                signal = np.pad(signal, (int(pad_len), int(pad_len) + int(pad_rem)), 'constant', constant_values=0)
            else:
                pad_len = s_len - mslen
                pad_len /= 2
                signal = signal[int(pad_len):int(pad_len + mslen)]
            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

            if flatten:
                # Flatten the datamslen = 32000
                mfcc = mfcc.flatten()
            data.append(mfcc)
            labels.append(i)
            cnt += 1
        print ("ended reading folder", directory)
        os.chdir('..')
    os.chdir(cur_dir)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
dataset_path = '/home/mpskkeerthu/Desktop/emotion_from_speech/dataset'
x_train, x_test, y_train, y_test = get_data(dataset_path=dataset_path, flatten=False)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print ('Starting CNN')
in_shape = x_train[0].shape
print(in_shape)
x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
input_shape=x_train[0].shape
num_classes=len(class_labels)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
p = np.random.permutation(len(x_train))
x_train = x_train[p]
y_train = y_train[p]
#model.fit(x_train, y_train, batch_size=32, epochs=1)
#los,acc=model.evaluate(x_test,y_test)
def createModel():
	model=Sequential()
	model.add(Conv2D(8, (13, 13),input_shape=(input_shape[0],input_shape[1], 1)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(Conv2D(8, (13, 13)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 1)))
	model.add(Conv2D(8, (13, 13)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(Conv2D(8, (3, 3)))
	model.add(BatchNormalization(axis=-1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 1)))
	model.add(Flatten())
	model.add(Dense(64))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(nClasses, activation='softmax'))
	return model
print ('Starting CNN')
model=createModel()
batch_size = 32
epochs = 50
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_test,y_test))
model.evaluate(x_test,y_test)
