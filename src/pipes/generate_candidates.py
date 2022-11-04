
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)
@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'pandas==1.3.5',
        'gcsfs',
        'fsspec',
        'google-cloud-aiplatform==1.18.1',
        'google-cloud-storage',
        'tensorflow==2.8',
        'tensorflow-hub==0.12.0',
        # 'tensorflow-estimator==2.8.0',
        # 'keras==2.8.0'
    ],
)
def generate_candidates(
    project: str,
    # run: str,
    location: str,
    version: str,
    bucket: str, 
    dest_folder: str,
    images_gcs_uri: str,
    emb_index_gcs_uri: str,
    index_json_name: str,
) -> NamedTuple('Outputs', [
    ('embedding_index_file_uri', str),
    ('embedding_index_gcs_dir', str),
    # ('saved_pretrained_model_gcs_location', str),
]):
    import os
    import os.path
    import time
    import logging
    import pandas as pd
    
    from google.cloud import aiplatform as vertex_ai
    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob
    
    
    from datetime import datetime
    import tensorflow as tf
    import tensorflow_hub as hub
    
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
    TF_HUB_MODEL_DIR = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    
    FILTER_INDEX_DIR = f'gs://{bucket}/indexes/{version}'
    
    IMG_PATH = f'gs://{bucket}/{dest_folder}/train/train'
    logging.info(f'IMG_PATH: {IMG_PATH}')
    LIST_DIR = tf.io.gfile.listdir(IMG_PATH)
    logging.info(f'Length of LIST_DIR: {len(LIST_DIR)}')
    
    CSV_URI = f'gs://{bucket}/{dest_folder}/train.csv'
    train_csv = pd.read_csv(CSV_URI)
    logging.info(f'CSV_URI: {CSV_URI}')
    
    vertex_ai.init(
        project=project,
        location=location,
    )
    
    # ==============================================================================
    # Define helper functions
    # ==============================================================================
    def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
        """Uploads a file to GCS bucket"""
        client = storage.Client(project=project)
        blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
        blob.bucket._client = client
        blob.upload_from_filename(source_file_name)
    
    def read_and_decode(filename, reshape_dims=[IMG_HEIGHT, IMG_WIDTH]):
        # Read the file
        img = tf.io.read_file(filename)

        # Convert the compressed string to a 3D uint8 tensor.
        img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
        img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

        # Resize the image to the desired size.
        return tf.image.resize(img, reshape_dims)
    
    def create_embeddings_dataset(embedder, img_path):
    
        dataset_embeddings = []
        ids_list = []
        id_cat_list = []

        start = time.time()

        list_dir = tf.io.gfile.listdir(img_path)
        for file in list_dir:
            img_tensor = read_and_decode(img_path + "/" + file, [IMG_WIDTH, IMG_HEIGHT])
            embeddings = embedder(img_tensor)

            IMAGE_ID = file.split(".")[0]
            CAT = train_csv.loc[train_csv['ImgId'] == IMAGE_ID, 'categories'].item()

            dataset_embeddings.extend(embeddings)
            ids_list.append(IMAGE_ID)
            id_cat_list.append(CAT)

        dataset_embeddings = tf.convert_to_tensor(dataset_embeddings)

        end = time.time()
        elapsed_time = round((end - start), 2)
        logging.info(f'Elapsed time writting embeddings: {elapsed_time} seconds\n')

        # return dataset_filenames, dataset_embeddings, ids_list, id_cat_list
        return dataset_embeddings, ids_list, id_cat_list
    
    # ==============================================================================
    # Download TF Hub model
    # ==============================================================================
    layers = [
        hub.KerasLayer(
            f"{TF_HUB_MODEL_DIR}",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            trainable=False,
            name='mobilenet_embedding'),
        tf.keras.layers.Flatten()
    ]
    model = tf.keras.Sequential(
        layers, name='visual_embedding'
    )
    
    # ==============================================================================
    # create embedding vectors
    # ==============================================================================
    dataset_embeddings, dataset_ids, dataset_id_cats = create_embeddings_dataset(
        lambda x: model.predict(x),
        IMG_PATH,
    )
    
    logging.info(f"Shape of embeddings dataset: {dataset_embeddings.shape}\n")
    logging.info(f"dataset_ids: {dataset_ids[:3]}\n")
    logging.info(f"dataset_id_cats: {dataset_id_cats[:3]}\n")
    
    cleaned_embs = [x.numpy() for x in dataset_embeddings] #clean up the output
    cleaned_id_cats = [x for x in dataset_id_cats] #clean up the output
    cleaned_ids = [x for x in dataset_ids] #clean up the output
    
    # ==============================================================================
    # write json index
    # ==============================================================================
    
    with open(f"{index_json_name}", "w") as f:
        counter = 0 
        for img_id, vector, img_cat in zip(cleaned_ids, cleaned_embs, cleaned_id_cats):
            f.write('{"id":"' + str(img_id) + '",')
            f.write('"embedding":[' + ",".join(str(x).strip() for x in list(vector)) + '],')
            f.write('"restricts":[{"namespace":"category","allow":["' + str(img_cat) + '"]},') #,
            f.write('{"namespace":"tag_1","allow":['+ ('"even"' if counter % 2 == 0 else '"odd"') + ']}]}\n')
            counter+=1
        f.close()
        
        
    _upload_blob_gcs(emb_index_gcs_uri, f"{index_json_name}", f"{index_json_name}")
    
    # embedding_index_file_uri = f'{FILTER_INDEX_DIR}/{index_json_name}'
    embedding_index_file_uri = f'{emb_index_gcs_uri}/{index_json_name}'
    logging.info("embedding_index_file_uri:", embedding_index_file_uri)

    return(
        f'{embedding_index_file_uri}',
        f'{emb_index_gcs_uri}', # 'gs://{BUCKET}/indexes/{VERSION}'
      # f'{save_path}',
    )
