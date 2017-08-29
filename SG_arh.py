""" Glavna mreža """
num_classes = 400
main_input = Input(shape=input_shape,name='image_part_input')

x = Conv2D(64, (3, 3), strides=(2, 2), padding="same",
           activation="relu")(main_input)
x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(128, (3, 3), strides=(2, 2), padding="same",
           activation="relu")(x)
x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(256, (3, 3), strides=(2, 2), padding="same",
           activation="relu")(x)
x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)

x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
main_output = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

""" Dodatni nivo k globalni mreži """
vgg16 = VGG16(weights="imagenet", include_top=True)
vgg_output = Dense(256, activation='softmax',
    name='predictions')(vgg16.layers[-2].output)

""" Združevanje glavne in globalne mreže """
def repeat_output(input):
    shape = K.shape(x)
    return K.reshape(K.repeat(input, 4 * 4), (shape[0], 4, 4, 256))

vgg_output = Lambda(repeat_output)(vgg_output)

# zamrzovanje nivojev mreže VGG16
for layer in vgg16.layers:
    layer.trainable = False

""" Združena mreža """
merged = concatenate([vgg_output, main_output], axis=3)

last = Conv2D(256, (3, 3), padding="same")(merged)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)
last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)
last = Conv2D(400, (3, 3), padding="same", activation="relu")(last)

def resize_image(x):
    return K.resize_images(x, 2, 2, "channels_last")

# večdimenzionalni softmax
def custom_softmax(x):
    sh = K.shape(x)
    x = K.reshape(x, (sh[0] * sh[1] * sh[2], num_classes))
    x = K.softmax(x)
    x = K.reshape(x, (sh[0], sh[1], sh[2], num_classes))
    return x

last = Activation(custom_softmax)(last)
last = Lambda(resize_image)(last)
