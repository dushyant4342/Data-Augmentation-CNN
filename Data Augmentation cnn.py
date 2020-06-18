#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[2]:


datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
            )


# In[3]:


img=load_img('Dushyant Profile Picture.jpg')


# In[4]:


x=img_to_array(img) #shape(3,150,150)
x=x.reshape((1,)+x.shape) #shape(1,3,150,150)


# In[5]:


i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='preview', save_prefix='dushyant',save_format='jpeg'):
    i+=1
    if i>20:
        break


# In[ ]:




