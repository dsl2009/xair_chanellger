import random
import tensorflow as tf
from data_set_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
import json
import os
#====================================================DEFINE YOUR ARGUMENTS=======================================================================
flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/new/step1', 'ss')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.0, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', 'plant', 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def main():


    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)

    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # First, convert the training and validation sets.

    
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    print(labels_to_class_names)
    #write_label_file(labels_to_class_names, FLAGS.dataset_dir)
    lb_f = os.path.join(FLAGS.dataset_dir,'labels.json')
    f = open(lb_f,'w')
    f.write(json.dumps(labels_to_class_names))
    f.flush()
    f.close()
    print ('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
    main()