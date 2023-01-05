#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


# In[ ]:


# Markdown format change

st.markdown('''
<style>
.main{
background-color: #FFFFE0;
}
<style>
''',
unsafe_allow_html= True)

# Markdown format change

st.sidebar.markdown('''
<style>
.main{
background-color: #ADD8E6;
}
<style>
''',
unsafe_allow_html= True)


# In[2]:


st.write('''
# Penguin Species Prediction App

This app uses random forest classifier to predict the ***Palmer Penguin*** species.

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
''')


# In[3]:


st.sidebar.header('User Input Features')

st.sidebar.markdown('''
[Example CSV input file](https://raw.githubusercontent.com/bhushan-b-borude/Streamlit_project_03_penguins_app/main/penguins_example.csv)
''')


# In[4]:


# Collect user input features into dataframe

uploaded_file = st.sidebar.file_uploader(' Upload your input file in CSV format',
                                        type=['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
        bill_length_mm = st.sidebar.slider('Bill length(mm)', 32.1, 59.6, 41.8)
        bill_depth_mm = st.sidebar.slider('Bill depth(mm)', 13.1, 21.5, 18.1)
        flipper_length_mm = st.sidebar.slider('Flipper length(mm)', 172.0, 231.0, 217.0)
        body_mass_g = st.sidebar.slider('Body mass(g)', 2700.0, 6300.0, 5000.0)
        
        data = {'island': island,
               'bill_length_mm': bill_length_mm,
               'bill_depth_mm': bill_depth_mm,
               'flipper_length_mm': flipper_length_mm,
               'body_mass_g': body_mass_g,
               'sex': sex.lower()}
        features_df = pd.DataFrame(data, index=[0])
        return features_df
    input_df = user_input_features() 


# In[5]:


# Combine user input features with entire penguins dataset
# This is import for feature encoding
# When we perform feature encoding on user input, we need all the possible options 
# which are available in entire penguins dataset

penguins = pd.read_csv('penguins_cleaned.csv')
penguins = penguins.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)


# In[6]:


# Encoding of ordinal features

encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] # Selecting only the first row(user input)


# In[7]:


# Display the user input featuers

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('CSV file is not uploaded. Currently using input features as below.')
    st.write(df)


# In[8]:


# Read in save classification model

load_rfc = pickle.load(open('penguins_rfc.pkl', 'rb'))


# In[9]:


# Apply model to make predictions

prediction = load_rfc.predict(df)
prediction_proba = load_rfc.predict_proba(df)


# In[10]:


species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])


# In[46]:


st.subheader('Penguin Species Prediction Probability')
pred_df = pd.DataFrame({'Species': species, 'Prediction probability': prediction_proba[0]})
st.write(pred_df)


# In[26]:


final_pred_proba = pred_df[pred_df['Species'] == species[prediction][0]].reset_index()['Prediction probability'][0]
st.subheader('Prediction')

st.write('We predict the species to be',species[prediction][0],
         'with prediction probability', final_pred_proba, '.')


# In[ ]:


if species[prediction][0] == 'Adelie':
    image = Image.open('adelie.jpg')
    source = 'Image Source: https://de.wikipedia.org/wiki/Adeliepinguin#/media/Datei:Adeliepinguin_in_der_Mauser_auf_Franklin_Island.jpg'
elif species[prediction][0] == 'Gentoo':
    image = Image.open('gentoo.jpg')
    source = 'Image Source: https://en.wikipedia.org/wiki/Gentoo_penguin#/media/File:Brown_Bluff-2016-Tabarin_Peninsula%E2%80%93Gentoo_penguin_(Pygoscelis_papua)_03.jpg'
elif species[prediction][0] == 'Chinstrap':
    image = Image.open('chinstrap.jpg')
    source = 'Image Source: https://upload.wikimedia.org/wikipedia/commons/6/69/Manchot_01.jpg'

    
st.write('Here is a photo of', species[prediction][0], 'Penguine for you!')
st.image(image, caption=source)


# In[ ]:


st.subheader('Reference')
st.markdown('[Code reference](https://github.com/dataprofessor/code/tree/master/streamlit/part3)')


# In[50]:


#pip freeze > penguins_app_requirements.txt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




