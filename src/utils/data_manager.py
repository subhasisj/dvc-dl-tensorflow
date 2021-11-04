import tensorflow as tf
import logging

def train_valid_generator(data_dir, IMAGE_SIZE, BATCH_SIZE, AUGMENTATION=False):
    """
    Generates batches of training and validation data using ImageDataGenerator. Perform Augmentation if AUGMENTATION is True.
    :param data_dir: Directory containing the training and validation data.
    :param IMAGE_SIZE,: Size of the images.
    :param BATCH_SIZE: Size of the batch.
    :param AUGMENTATION: Boolean value indicating whether to perform data augmentation.
    :return: Training and validation batches.
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset="training",
        shuffle=True,
        seed=42
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset="validation",
        shuffle=False,
        seed=42
    )
    if not AUGMENTATION:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255,
                validation_split=0.2
            )

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset="training",
            shuffle=True,
            seed=42
        )
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset="validation",
            shuffle=False,
            seed=42
        )
    logging.info("Data generators created.")
    return train_generator, validation_generator