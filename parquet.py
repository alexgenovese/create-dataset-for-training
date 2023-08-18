import pandas as pd
from PIL import Image
import io
import os

# Start reading parquet file
df_dataset = pd.read_parquet('/content/train-00000-of-00001-566cc9b19d7203f8.parquet', engine='fastparquet')

# create captions from parquet columns
# text, image.bytes, image.path 
df = []
for i in range(len(df_dataset)):
    image = Image.open(io.BytesIO(df_dataset['image.bytes'].iloc[i]))
    img_path = os.path.join(dir_name, f'{i}.jpg')
    image.save(img_path)
    df.append([img_path, df_dataset['text'].iloc[i]])
df = pd.DataFrame(df)
df.columns = ['paths', 'caption']
df.to_csv('captions.csv', index=False)


