import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np

train_data_dir = '/home/chillmate/Desktop/ImgPro/ImgFood/train/meat'
test_data_dir = '/home/chillmate/Desktop/ImgPro/ImgFood/test/meat'

batch_size = 64

def resize_image(img):
    img_resized = tf.image.resize(img, (244, 244))
    return img_resized

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=resize_image
)

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=resize_image)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(244, 244),  
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(244, 244),  
    batch_size=batch_size,
    class_mode='categorical'
)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    filepath='groupMeat_class_epoch_{epoch:02d}.h5',
    save_freq='epoch',
    period=100
) 

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=300,
    callbacks=[checkpoint_callback],
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

class_labels = list(test_generator.class_indices.keys())

predicted_labels = [class_labels[idx] for idx in predicted_classes]

print("Predicted Labels:", predicted_labels)


# -----------------------------------------------------------------------------------------------------test

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# model = load_model('/home/chillmate/Desktop/ImgPro/Lib/groupMeat_class_epoch_200.h5')
# img_path = '/home/chillmate/Desktop/ImgPro/ImgFood/train/meat/Seafood/squid/Image_4.jpg'
# img = image.load_img(img_path, target_size=(244, 244))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0) 
# img_array /= 255. 

# predictions = model.predict(img_array)
# sorted_indices = np.argsort(-predictions[0])
# class_labels = ['OtherMeats and Mushroom', 'Poultry', 'Seafood']

# meat_groups = {
#     'Poultry': ['chicken', 'duck'],
#     'Seafood': ['crab', 'crayfish', 'fish', 'shellfish', 'shrimp', 'squid'],
#     'OtherMeats and Mushroom': ['beef', 'egg', 'lamb', 'mushroom', 'offal', 'pork','Golden needle mushroom']
# }

# print("Predicted Labels (arranged by confidence):")
# for idx, ordinal in enumerate(sorted_indices, start=1):
#     predicted_label = class_labels[ordinal]
#     confidence = predictions[0][ordinal]
#     print(f"{idx}. Label: {predicted_label} - Confidence: {confidence}")
#     subclass = meat_groups[predicted_label]
#     sorted_subclass = sorted(subclass, key=lambda x: (predictions[0][class_labels.index(predicted_label)], x.lower()))
#     print(f"   Subclass (arranged by confidence and alphabetically):")
#     for sub_idx, sub_name in enumerate(sorted_subclass, start=1):
#         sub_confidence = predictions[0][class_labels.index(predicted_label)] if sub_name in meat_groups[predicted_label] else 0
#         print(f"      {sub_idx}. {sub_name} - Confidence: {sub_confidence * 100:.2f}%")