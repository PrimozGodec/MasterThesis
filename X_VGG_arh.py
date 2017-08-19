vgg16 = VGG16(weights="imagenet", include_top=False,
              input_shape=input_shape)

# freeze vgg16
for layer in vgg16.layers:
    layer.trainable = False

last = UpSampling2D(size=(2, 2))(vgg16.output)
last = Conv2D(256, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)
last = Conv2D(256, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(128, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)
last = Conv2D(128, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(64, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)
last = Conv2D(64, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)

last = UpSampling2D(size=(2, 2))(last)
last = Conv2D(32, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)
last = Conv2D(2, (3, 3), padding="same",
              activation="relu",
              kernel_regularizer=regularizers.l2(
                  0.01))(last)


def resize_image(x):
    return K.resize_images(x, 2, 2, "channels_last")


last = Lambda(resize_image)(last)
