""" Glavna mreža """
main_input = Input(shape=input_shape, name='image_part_input')

x = Conv2D(64, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01),
           name="conv1")(main_input)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x1 = Conv2D(128, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            name="conv2")(x)
x = Conv2D(128, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01),
           name="conv3")(x1)
x = Conv2D(128, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01),
           name="conv4")(x)
x = add([x, x1])

x = Conv2D(128, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01),
           name="conv5")(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x1 = Conv2D(256, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(0.01),
            name="conv6")(x)
x = Conv2D(256, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01),
           name="conv7")(x1)
x = Conv2D(256, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01),
           name="conv8")(x)
x = add([x, x1])

x = Conv2D(256, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01))(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x1 = Conv2D(512, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(0.01))(
    x)
x = Conv2D(512, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01))(
    x1)
x = Conv2D(512, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01))(x)
x = add([x, x1])

x = Conv2D(512, (3, 3), padding="same", activation="relu",
           kernel_regularizer=regularizers.l2(0.01))(x)
main_output = Conv2D(
    256, (3, 3), padding="same", activation="relu",
    kernel_regularizer=regularizers.l2(0.01))(x)

""" Dodatni nivo k globalni mreži """
vgg16 = VGG16(weights="imagenet", include_top=True)
vgg_output = Dense(256, activation='relu',name='predictions')(
    vgg16.layers[-2].output)

""" Združevanje glavne in globalne mreže """
def repeat_output(input):
    shape = K.shape(x)
    return K.reshape(K.repeat(input, 28 * 28),
                     (shape[0], 28, 28, 256))


vgg_output = Lambda(repeat_output)(vgg_output)

# zamrzovanje nivojev mreže VGG16
for layer in vgg16.layers:
    layer.trainable = False

merged = concatenate([vgg_output, main_output], axis=3)

""" Združena mreža """
last = Conv2D(128, (3, 3), padding="same")(merged)

last = Conv2DTranspose(
    64, (3, 3), strides=(2, 2), padding="same",
    activation="relu",
    kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(64, (3, 3), padding="same", activation="relu",
              kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(64, (3, 3), padding="same", activation="relu",
              kernel_regularizer=regularizers.l2(0.01))(last)

last = Conv2DTranspose(
    64, (3, 3), strides=(2, 2), padding="same",
    activation="relu",
    kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(32, (3, 3), padding="same",activation="relu",
              kernel_regularizer=regularizers.l2(0.01))(last)
last = Conv2D(2, (3, 3), padding="same",activation="relu",
              kernel_regularizer=regularizers.l2(0.01))(last)


def resize_image(x):
    return K.resize_images(x, 2, 2, "channels_last")


last = Lambda(resize_image)(last)
