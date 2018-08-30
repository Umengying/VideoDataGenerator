from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import backend as K
import numpy as np
import os
import math
import multiprocessing.pool


class VideoDirIterator(image.Iterator):
    def __init__(self,
                 directory,
                 video_data_generator,
                 target_size=(112,112),
                 batch_size=32,
                 clip_size=16,
                 shuffle=True,
                 allow_lt_clip_size=True,
                 seed=None,
                 data_format="channels_last",
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        # TODO: parammeter explaination
        """
        directory:
        video_data_generator:
        target_size:
        batch_size:
        clip_size:
        shuffle:
        allow_lt_clip_size: whether allow the number of frames less than clip size
        seed:
        data_format: "channel_last": "NDHWC", "channel_first": "NCDHW".
        save_to_dir:
        save_prefix:
        save_format:
        """
        if data_format is None:
            data_format = K.image_data_format
        self.directory = directory
        self.video_data_generator = video_data_generator
        self.target_size = tuple(target_size)
        self.batch_size = batch_size
        self.clip_size = clip_size
        self.allow_lt_clip_size = allow_lt_clip_size
        self.data_format = data_format
        # set video shape
        if self.data_format == 'channels_last':
            self.video_shape = (self.clip_size,) + self.target_size + (3,)
        else:
            self.video_shape = (3,) + (self.clip_size,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.class_name = self._get_class_name()
        self.num_classes = len(self.class_name)
        self.class_indices = dict(zip(self.class_name, range(self.num_classes)))

        self.filenames, self.classes = self._get_filenames_and_classes()
        self.samples = len(self.filenames)

        print('Found %d videos belonging to %d classes.' % (self.samples, self.num_classes))

        super(VideoDirIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # "NDHWC": channel last
        # "NCDHW": channel first
        batch_shape = (len(index_array),) + self.video_shape
        batch_x = np.zeros(batch_shape, dtype=K.floatx())
        batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # get filename, regard the filenames as video dir full path
            vfname = self.filenames[j]
            frame_list = os.listdir(vfname)
            # get list of frames, and choose clip_size of frames
            if self.allow_lt_clip_size and len(frame_list)<=self.clip_size:
                # repeat the frame list, if the length of list is less than clip size
                num = math.ceil(self.clip_size/len(frame_list))
                batch_list = num * frame_list
                batch_list = batch_list[0 : self.clip_size]
            else:
                index = np.random.randint(0, len(frame_list)-self.clip_size)
                batch_list = frame_list[index : index+self.clip_size]
            # transform
            # TODO: apply the same transformation to a clip or different transformation to each frame?
            # currently, I determin to apply the same transformation to a clip
            clip = []
            for frame in batch_list:
                img_path = os.path.join(vfname, frame)
                img = image.load_img(img_path, target_size=self.target_size)
                # convert image to array, set "channels_last" as the default data format
                # in the end convert the data format to users' configuraton
                x = image.img_to_array(img, data_format="channels_last")
                x = self.video_data_generator.random_transform(x)
                x = self.video_data_generator.standardize(x)
                clip.append(x)

            # generate batch_x
            clip = np.array(clip)
            if self.data_format == 'channels_first':
                clip = clip.transpose((3, 0, 1, 2))
            batch_x[i] = clip
            # generate batch_y
            labels = np.array(self.classes)
            for i, label in enumerate(labels[index_array]):
                batch_y[i, label] = 1

            # TODO: additional function

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        Returns:
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_class_name(self):
        # traverse root dir
        class_name = []
        for subdir in sorted(os.listdir(self.directory)):
            if os.path.isdir(os.path.join(self.directory, subdir)):
                class_name.append(subdir)
        return class_name

    def _get_filenames_and_classes(self):
        filenames = []
        results = []
        # classes = np.zeros((self.samples,), dtype='int32')
        classes = []
        i = 0
        pool = multiprocessing.pool.ThreadPool()

        for dirpath in (os.path.join(self.directory, subdir)
                        for subdir in self.class_name):
            results.append(
                pool.apply_async(self._list_video_samples, (dirpath,))
            )

        for res in results:
            filename, cls = res.get()
            # classes[i:i+len(cls)] = cls
            classes = classes + cls
            filenames += filename
            i += len(cls)

        pool.close()
        pool.join()
        return filenames, classes

    def _list_video_samples(self, base_path):
        filenames = []
        classes = []
        dir_name = os.path.basename(base_path)

        for subdir in sorted(os.listdir(base_path)):
            filename = os.path.join(base_path, subdir)
            if not os.path.isdir(filename):
                continue
            if self.allow_lt_clip_size:
                filenames.append(filename)
                classes.append(self.class_indices[dir_name])
            else:
                if self._list_frame(filename):
                    filenames.append(filename)
                    classes.append(self.class_indices[dir_name])

        return filenames, classes

    def _list_frame(self, base_path):
        if len(os.listdir(base_path)) < self.clip_size:
            return False
        else:
            return True


class VideoDataGenerator(image.ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format="channels_last",
                 validation_split=0.0):
        super(VideoDataGenerator, self).__init__(
                 featurewise_center=featurewise_center,
                 samplewise_center=samplewise_center,
                 featurewise_std_normalization=featurewise_std_normalization,
                 samplewise_std_normalization=samplewise_std_normalization,
                 zca_whitening=zca_whitening,
                 zca_epsilon=zca_epsilon,
                 rotation_range=rotation_range,
                 width_shift_range=width_shift_range,
                 height_shift_range=height_shift_range,
                 brightness_range=brightness_range,
                 shear_range=shear_range,
                 zoom_range=zoom_range,
                 channel_shift_range=channel_shift_range,
                 fill_mode=fill_mode,
                 cval=cval,
                 horizontal_flip=horizontal_flip,
                 vertical_flip=vertical_flip,
                 rescale=rescale,
                 preprocessing_function=preprocessing_function,
                 data_format=data_format,
                 validation_split=validation_split)

    def flow_from_directory(self,
                            directory,
                            target_size=(112, 112),
                            batch_size=32,
                            clip_size=16,
                            allow_lt_clip_size=True,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            ):
        return VideoDirIterator(
            directory,
            self,
            target_size=target_size,
            batch_size=batch_size,
            clip_size=clip_size,
            allow_lt_clip_size=allow_lt_clip_size,
            shuffle=shuffle,
            seed=None,
            data_format=self.data_format,
            save_to_dir=None,
            save_prefix='',
            save_format='png')
