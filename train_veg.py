import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
train_data_dir = '/home/chillmate/Desktop/ImgPro/ImgFood/train/vegetable'
test_data_dir = '/home/chillmate/Desktop/ImgPro/ImgFood/test/vegetable'
batch_size = 32
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
    class_mode='sparse',
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    filepath='groupVeg_class_epoch_{epoch:02d}.h5',
    save_freq='epoch',
    period=100
) 

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=500,
    callbacks=[checkpoint_callback],
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted Classes:", predicted_classes)


# --------------------------------------------------------------------test


# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# model = load_model('/home/chillmate/Desktop/ImgPro/Lib/groupVeg_class_epoch_200.h5')
# img_path = '/home/chillmate/Desktop/ImgPro/ImgFood/test/vegetable/White and light-colored vegetables/cauliflower/Image_2.jpg'
# img = image.load_img(img_path, target_size=(244, 244))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.
# predictions = model.predict(img_array)
# sorted_indices = np.argsort(-predictions[0])
# class_labels = ['Green vegetables','Red and orange vegetables','White and light-colored vegetables']
# vegetable_groups = {
#     'Green vegetables': ['Basil', 'Spring onion', 'Pepper', 'Coriander', 'Lemon grass', 'Ivy gourd', 'Sweet basil', 'Water spinach', 'Acacia', 'Long beans', 'Spinach', 'Broccoli', 'Kaffir lime', 'Peppermint', 'Chinese cabbage', 'Cucumber', 'Kale', 'Salad vegetables', 'Cabbage', 'Bitter melon', 'Asparagus', 'Squash', 'Neem', 'Eggplant', 'Bitter bean', 'Celery', 'Bok choy', 'Garlic chives'],
#     'Red and orange vegetables': ['Onion', 'Carrot', 'Chilli pepper', 'Capsicum', 'Bell pepper'],
#     'White and light-colored vegetables': ['Bamboo shoot', 'Ginger', 'Cauliflower', 'Lotus stem', 'Baby corn', 'Bean sprout', 'Raddish', 'Garlic', 'Sesbania grandiflora'],
#     }
# print("Predicted Labels (arranged by confidence):")
# for idx, ordinal in enumerate(sorted_indices, start=1):
#     predicted_label = class_labels[ordinal]
#     confidence = predictions[0][ordinal]
#     print(f"{idx}. Label: {predicted_label} - Confidence: {confidence}")
#     subclass = vegetable_groups[predicted_label]
#     sorted_subclass = sorted(subclass, key=lambda x: (predictions[0][class_labels.index(predicted_label)], x.lower()))
#     print(f"   Subclass (arranged by confidence and alphabetically):")
#     for sub_idx, sub_name in enumerate(sorted_subclass, start=1):
#         sub_confidence = predictions[0][class_labels.index(predicted_label)] if sub_name in vegetable_groups[predicted_label] else 0
#         print(f"      {sub_idx}. {sub_name} - Confidence: {sub_confidence * 100:.2f}%")