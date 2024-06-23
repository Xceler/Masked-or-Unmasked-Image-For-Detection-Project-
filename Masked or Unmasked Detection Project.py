import zipfile 

zip_fil = 'face-mask-dataset.zip'
with zipfile.ZipFile(zip_fil, 'r') as fil:
    fil.extractall() 

import os
import shutil 
import random 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization 



#Split file dir into train and val dir 

def split_data(source_dir, train_dir, val_dir, split_ratio = 0.2):
    for category in ['simple', 'complex']:
        category_path = os.path.join(source_dir, category)

        train_category_path = os.path.join(train_dir, category)
        val_category_path = os.path.join(val_dir, category)

        #Create train and val dataset 
        os.makedirs(train_category_path, exist_ok = True)
        os.makedirs(val_category_path, exist_ok = True)

        all_fil = os.listdir(category_path)
        random.shuffle(all_fil)

        split_index = int(len(all_fil) * (1 - split_ratio))

        train_fil = all_fil[:split_index]
        val_fil = all_fil[split_index:]
        
        for fil in train_fil:
            shutil.copy(os.path.join(category_path, fil), os.path.join(train_category_path, fil))

        for fil in val_fil:
            shutil.copy(os.path.join(category_path, fil), os.path.join(val_category_path, fil))
        


def main():
    masked_dir = 'FMD_DATASET/with_mask'
    unmasked_dir = 'FMD_DATASET/without_mask'

    train_dir = "Train_Dir"
    val_dir = 'Val_Dir'

    train_masked_dir = os.path.join(train_dir, 'masked')
    val_masked_dir = os.path.join(val_dir, 'masked')

    train_unmasked_dir = os.path.join(train_dir, 'unmasked')
    val_unmasked_dir = os.path.join(val_dir, 'unmasked')

    split_data(masked_dir, train_masked_dir, val_masked_dir)
    split_data(unmasked_dir, train_unmasked_dir, val_unmasked_dir)


if __name__ == '__main__':
    main() 


#Data Preprocessing 

train_dir = 'Train_Dir'
val_dir = 'Val_Dir'

train_gen = ImageDataGenerator(
    rescale = 1./255.0,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.5,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

val_gen = ImageDataGenerator(
    rescale = 1./255.0,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.5,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_datagen = train_gen.flow_from_directory(
    train_dir, 
    target_size = (228, 228),
    batch_size = 32,
    class_mode = 'binary'
)

val_datagen = val_gen.flow_from_directory(
    val_dir, 
    target_size = (228, 228),
    batch_size = 32,
    class_mode = 'binary'
)


#Model Architecture Deep Learning Building 

model = Sequential([

    Conv2D(32, (3,3), activation = 'relu', input_shape = (228, 228, 3)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(128, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(256, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Flatten(),

    Dense(1024, activation = 'relu'),
    BatchNormalization(),

    Dense(1024, activation = 'relu'),
    BatchNormalization(),
    
    Dense(1, activation = 'softmax')
])


#Compiling the model 
model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',
              metrics = ['accuracy'])

        
model.fit(train_datagen, epochs = 1, validation_data = val_datagen)

def predict_img(model, img_path):
    img = load_img(img_path, target_size = (228, 228))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis   = 0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predictions = prediction[0]

    if predictions > 0.5:
        print('Unmasked')
    else:
        print('Masked')



img_path = 'Train_Dir/unmasked/simple/simple12.jpg'
result = predict_img(img_path)
print("Result:", result)
