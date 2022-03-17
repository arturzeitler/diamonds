import os
import zipfile
import logging
import sys
import shutil

import boto3
import botocore
import numpy as np
import pandas as pd
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

log.info("Logger created")

# doanload data

bucket_name = 'diamonds-ml-project' 
csv_name = 'diamonds2.csv' 
dataset_fldr = os.path.join('./', 'dataset')
zip_model = 'model.zip'
model_flag = None
saved_serve_path = os.path.join('./diamonds', '0001')
checkpoint_path = os.path.join('./checkpoints', 'best_epoch_weights.ckpt')
log.info("starting download")

try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ['aws_access_key_id'],
        aws_secret_access_key=os.environ['aws_secret_access_key']
    )
except KeyError:
    s3_client = boto3.client(
        's3',
    )

with open(os.path.join(dataset_fldr, csv_name), 'wb') as data:
    s3_client.download_fileobj(bucket_name, csv_name, data)

try:
    with open(os.path.join(dataset_fldr, csv_name), 'wb') as data:
        s3_client.download_fileobj(bucket_name, csv_name, data)
    log.info('Found csv on S3')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        log.info('Bucket ' + bucket_name + ' with key ' + csv_name + ' does not exist')
    else:
        raise

try:
    with open('./model.zip', 'wb') as data:
        s3_client.download_fileobj(bucket_name, zip_model, data)
    model_flag = 1
    with zipfile.ZipFile(os.path.join(zip_model),"r") as zip_ref:
        zip_ref.extractall('./')
    log.info('Found trained model')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        log.info('Bucket ' + bucket_name + ' with key ' + zip_model + ' does not exist')
    else:
        raise

log.info("download successful")

# preprocess data for model
SEED = 23
tf.random.set_seed(
    seed=SEED
)

df = pd.read_csv(os.path.join(dataset_fldr, csv_name)).drop(['Unnamed: 0'], axis=1)
df_val = df.sample(frac=0.15, random_state=SEED)
df_train = df.drop(df_val.index, errors="ignore")
x_train, y_train = df_train.drop(['price'], axis=1), df_train['price']
x_val, y_val = df_val.drop(['price'], axis=1), df_val['price']

inputs = {}

for col in x_train.columns:
    dtype = x_train[col].dtypes
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    inputs[col] = tf.keras.Input(shape=(1,), name=col, dtype=dtype)

train_features_dict = {name: np.array(value)
                         for name, value in x_train.items()}

val_features_dict = {name: np.array(value)
                         for name, value in x_val.items()}

train_ds = tf.data.Dataset.from_tensor_slices((train_features_dict, df_train['price']))
val_ds = tf.data.Dataset.from_tensor_slices((val_features_dict, df_val['price']))

train_ds_batches = train_ds.shuffle(len(df_train['price'])).batch(32)
val_ds_batches = val_ds.shuffle(len(df_val['price'])).batch(32)

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = tf.keras.layers.Normalization()
norm.adapt(np.array(x_train[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(x_train[name]))
    one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)

dia_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

# define model

def dia_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)
    return model

if model_flag:
    dia_model = tf.keras.models.load_model('./full_model.h5')
    optimizer = dia_model.optimizer
    log.info('Trained model and optimizer initialized')
else:
    dia_model = dia_model(dia_preprocessing, inputs)
    init_lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
    log.info('New model and optimizer instantiated')

epochs = 5
loss_fn = tf.keras.losses.MeanSquaredError()
train_acc_metric = tf.keras.metrics.MeanSquaredError()
val_acc_metric = tf.keras.metrics.MeanSquaredError()
temp_val_acc = np.inf

# train
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = dia_model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, dia_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, dia_model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = dia_model(x, training=False)
    val_loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)
    return val_loss_value

for epoch in range(epochs):
    log.info("\nStart of epoch %d" % (epoch + 1,))

    # Iterate over the training batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds_batches):
        loss_value = train_step(x_batch_train, y_batch_train)

    train_acc = train_acc_metric.result()
    log.info("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_ds_batches:
        val_loss_value = test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()

    val_acc_metric.reset_states()
    log.info("Validation acc: %.4f" % (float(val_acc),))
    if temp_val_acc > val_acc:
        log.info('New validation acc %.4f better than prior best %.4f' % (float(val_acc), float(temp_val_acc)))
        temp_val_acc = val_acc
        dia_model.save_weights(checkpoint_path)
        log.info('saved!')
    else:
        log.info('Validation acc did not improve - best %.4f' % (float(temp_val_acc)))

# save model & weights
dia_model.load_weights(checkpoint_path)
dia_model.save(saved_serve_path)

dia_model.compile(loss=loss_fn, optimizer=optimizer)
tf.keras.models.save_model(dia_model, filepath='./full_model.h5')

shutil.make_archive('weights', 'zip', saved_serve_path)

with zipfile.ZipFile('./model.zip', 'w') as zipf:
    zipf.write('./full_model.h5')

# send to s3

with open('./weights.zip', "rb") as f:
    s3_client.upload_fileobj(f, bucket_name, 'weights.zip')

with open('./model.zip', "rb") as f:
    s3_client.upload_fileobj(f, bucket_name, 'model.zip')
