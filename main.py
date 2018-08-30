from keras_dataset import VideoDataGenerator


if __name__ == '__main__':

    gen = VideoDataGenerator()
    data_gen = gen.flow_from_directory(
        directory="/data1/liangkai.yu/dataset/UCF101",
        target_size=(112,112),
        allow_lt_clip_size=True,
        shuffle=True
    )

    i = 0
    while i <= data_gen.samples // data_gen.batch_size:
        x, y = next(data_gen)
        i += 1
        print("{} -- {} -- {}".format(i, x.shape, y.shape))

    print(data_gen.samples)
