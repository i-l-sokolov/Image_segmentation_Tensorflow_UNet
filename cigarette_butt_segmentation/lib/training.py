import os
import tensorflow as tf
from model import get_model
from dataset import get_dataset
import argparse

parser = argparse.ArgumentParser(description='Parameters for training model: number of epochs and access to several GPUs')

#Add arguments with default values
parser.add_argument('--epochs', type=int, default=50, help='The number of epoch for training')
parser.add_argument('--mirror', type=bool, default=False, help='if device has several GPUs then mirror strategy distributes training among it')
parser.add_argument('--save_ds', type=bool, default=True, help='save datasets after creating for speed up in next training')
parser.add_argument('--batch', type=int, default=10, help='the size of batch during training')

# Parse arguments
args = parser.parse_args()

epochs = args.epochs
mirror = args.mirror
save_ds = args.save_ds
batch_value = args.batch

class Training():
    """
    Class for training and
    """
    def __init__(self, model, batch, train_ds, val_ds):
        """

        Parameters
        ----------
        model
        batch
        train_ds
        val_ds
        """
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch = batch
        self.best_weights = self.model.get_weights()  # The best wieghts will be replaced during traing
        self.best_epoch = 0


    def augmentation(self, image, mask):
        """
        Doc
        """
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=0.2)
        #         mask = tf.expand_dims(mask, -1)
        conc = tf.concat([image, mask], axis=-1)
        conc = tf.image.random_flip_left_right(conc)
        conc = tf.image.random_flip_up_down(conc)
        return conc[:, :, :3], conc[:, :, 3:]

    def training(self, n_epochs=epochs):
        """
        Doc
        """
        #         self.val_ds = self.val_ds.batch(self.batch)
        self.val_loss = self.model.evaluate(self.val_ds.batch(self.batch))[0]
        #         self.train_ds = self.train_ds.batch(self.batch)
        self.best_epoch = 0
        for epoch in range(n_epochs):
            if epoch == 0:
                history = self.model.fit(self.train_ds.batch(self.batch),
                                         validation_data=self.val_ds.batch(self.batch),
                                         epochs=1)
            elif epoch > 0:
                history = self.model.fit(self.train_ds.shuffle(500).map(self.augmentation).batch(self.batch),
                                         validation_data=self.val_ds.batch(self.batch),
                                         epochs=1)
            if history.history['val_loss'][0] < self.val_loss:
                self.val_loss = history.history['val_loss'][0]
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch + 1
            else:
                self.model.set_weights(self.best_weights)
        #         self.model.set_weights(self.best_weights)
        print(f'Best results was achieved on epoch {self.best_epoch} with val_loss {self.val_loss}')


def open_dataframes():
    """
    Opening dataframes
    Returns
    -------
    train_ds, val_ds
    """
    if os.path.exists('trains_ds') and os.path.exists('val_ds'):
        tr_ds = tf.data.Dataset.load('train_ds')
        v_ds = tf.data.Dataset.load('val_ds')
    else:
        tr_ds = get_dataset('train').with_options(options)
        v_ds = get_dataset('val').with_options(options)
        if save_ds:
            tf.data.Dataset.save(tr_ds, 'train_ds')
            tf.data.Dataset.save(v_ds, 'val_ds')
    return tr_ds, v_ds


def getting_training_class(tr_ds, v_ds):
    """

    Parameters
    ----------
    tr_ds - training dataset
    v_ds - validation dataset

    Returns
    -------
    class for training
    """
    tr_class = Training(get_model(loss_fn='binary_crossentropy', activation_last_layer='sigmoid'),
                                batch=batch_value,
                                train_ds=tr_ds,
                                val_ds=v_ds)
    return tr_class


if __name__ == '__main__':

    if mirror:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])
        with mirrored_strategy.scope():
            #getting datasets without batch
            train_ds, val_ds = open_dataframes()

            train_ds = train_ds.with_options(options)
            val_ds = val_ds.with_options(options)
            # getting class Training
            unet_tr = getting_training_class(train_ds,val_ds)
    else:
        train_ds, val_ds = open_dataframes()
        unet_tr = getting_training_class(train_ds,val_ds)

    #Training model and saving the best results
    unet_tr.training()
    unet_tr.model.save(f'../model/unet_{epochs}epochs.h5')
