#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install google-cloud-automl


# In[19]:


from google.cloud import automl


# In[20]:


from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)


# In[21]:


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/YOURFILEPATH/googleautomlcredentials.json" 
#need to generate automl credentials json


# In[ ]:


# TODO(developer): Uncomment and set the following variables
# project_id = "YOUR_PROJECT_ID"
# model_id = "YOUR_MODEL_ID"
# input_uri = "gs://YOUR_BUCKET_ID/path/to/your/input/csv_or_jsonl"
# output_uri = "gs://YOUR_BUCKET_ID/path/to/save/results/"

prediction_client = automl.PredictionServiceClient()

# Get the full path of the model.
model_full_id = f"projects/{project_id}/locations/us-central1/models/{model_id}"

gcs_source = automl.GcsSource(input_uris=[input_uri])

input_config = automl.BatchPredictInputConfig(gcs_source=gcs_source)
gcs_destination = automl.GcsDestination(output_uri_prefix=output_uri)
output_config = automl.BatchPredictOutputConfig(
    gcs_destination=gcs_destination
)

response = prediction_client.batch_predict(
    name=model_full_id,
    input_config=input_config,
    output_config=output_config
)

print("Waiting for operation to complete...")
print(
    f"Batch Prediction results saved to Cloud Storage bucket. {response.result()}"
)

