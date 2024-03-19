from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


train_path = "train"
test_path = "test"

train_datage = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.08,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    horizontal_flip=True,
    fill_mode="nearest",
)


test_datage = ImageDataGenerator(rescale=1 / 255.0)

train_batches = train_datage.flow_from_directory(
    "train",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=True,
)

test_batches = test_datage.flow_from_directory(
    "test",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=True,
)

# for my_batch in train_batches:
#     images = my_batch[0]
#     labels = my_batch[1]
#     for i in range(len(labels)):
#         plt.imshow(images[i], cmap="gray")
#         plt.show()
#         print(labels[i])
#     break

images, labels = next(train_batches)


model = Sequential(
    [
        Conv2D(
            filters=32, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 1)
        ),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


model.fit(
    train_batches,
    epochs=20,
    validation_data=test_batches,
)

images, labels = next(test_batches)


score = model.evaluate(images, labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


model.save("trained_model")
