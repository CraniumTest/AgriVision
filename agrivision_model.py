import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Image Data Preparation
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
img_height, img_width = 224, 224  # MobileNetV2 standard input size

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_data_dir, target_size=(img_height, img_width),
    batch_size=32, class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir, target_size=(img_height, img_width),
    batch_size=32, class_mode='categorical')

# Model Building
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assuming 3 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the Model
model.fit(train_generator, epochs=10, validation_data=validation_generator, verbose=1)

# Save the Model
model.save('agrivision_crop_model.h5')

# Prediction Function
def predict_crop_health(image_path):
    from keras.preprocessing import image
    import numpy as np

    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    classes = ['healthy', 'pest-infected', 'nutrient-deficient']
    
    return classes[np.argmax(prediction)]

# Usage Example
# print(predict_crop_health('some_image.jpg'))
