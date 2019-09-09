python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record --image_dir=images

python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record --image_dir=images
