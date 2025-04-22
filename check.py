import tensorflow as tf

tfrecord_file = "dataset.tfrecord"
dataset = tf.data.TFRecordDataset(tfrecord_file)

count = 0
bad_records = 0

for raw_record in dataset:
    try:
        _ = tf.train.Example.FromString(raw_record.numpy())  # Try parsing
        count += 1
    except Exception as e:
        bad_records += 1
        print(f"Bad record at index {count}: {e}")

print(f"\nTotal Records Checked: {count}")
print(f"Corrupted Records Found: {bad_records}")
