import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

st.title('Analysis of telecom data')
st.write("Here's our first attempt at using data to create a table:")

with dataset:
	st.header('telecom dataset')
	st.text('I gathered data from users ability to acess data and network ')

	text_data = pd.read_csv('processed_telecom.csv')
	st.write(text_data.head())

	st.subheader('Maximum number of uer interaction')
	User_interaction = pd.DataFrame(text_data['No_of_xDRsessions'].value_counts())
	st.bar_chart(User_interaction)

	st.subheader('Total number of data usage')
	User_interactionsp = pd.DataFrame(text_data['Total_MB'].value_counts()).head(50)
	st.bar_chart(User_interactionsp)

with features:
	st.header('The features I created')
	st.markdown('* ** The first features I created were about data access on different applications')
	st.markdown('* ** The second features I created were about categorising data')

with model_training:
	st.header('Time to train the model!')
	st.text('Here is my model that was used to train our dataset using Random Forest Classifier')
	sel_col, disp_col = st.beta_columns(2)
	#sel_col.slider('What should be the maximum depth pof model?' min_value= 10, max_value=20, step=10)
	max_depth = sel_col.slider('What should be the maximum depth pof model?', min_value= 10, max_value=100, value=20, step=10)
	n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index = 0)
	
	sel_col.text('Here is a list of features in my data:')
	sel_col.write(text_data.columns)

	input_feature = sel_col.text_input('which feature is used as input?','No_of_xDRsessions')
	
	if n_estimators =='No limit':
		regr =  RandomForestRegressor(max_depth=maxdepth)
	else:
		regr =  RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

	#regr = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
	
	X = text_data[[input_feature]]
	y =text_data[['trip_distance']]

	regr.fit(X,y)
	predictio = regr.predict(y)

	disp_col.subheader('Mean absolute error of the model is:')
	disp_col.write(mean_absolute_error(y, prediction))

	disp_col.subheader('Mean squared error of the model is:')
	disp_col.write(mean_squared_error(y, prediction))

	disp_col.subheader('R squared score error of the model is:')
	disp_col.write(r2, score_error(y, prediction))
