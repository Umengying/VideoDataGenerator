# VideoDataGenerator
An Keras implementation of video data generator for action recognition

## Introduction
By inheriting the `ImageDataGenerator` class, I implemented `VideoDataGenerator` to read video data for video classification.

## Dependency

- python3
- TensorFlow 1.8

## Usage

#### First, make sure your dataset directory structure looks like the one described below.
Taking UCF101 dataset for example. UCF101 is the first level dir, it should contain subdirectory for each class, so there should be 101 subdirectories. Each subdirectory contains many video data folders, each video data folder contains a sequence of frames that converted from video file using ffmepg.

#### Second, using `VideoDataGenerator` as you use `ImageDataGenerator`

## Note

- VideoDataGenerator is based on `tf.keras` not `keras`
- Some functions are not supported now, such as `flow()`