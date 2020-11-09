# UsedCarPricePredictor
A Python application which can use a variety of ML techniques to predict the value of a vehicle based on several criteria.

# Prerequisites
Requires Python - https://www.python.org/downloads/release/python-390/
  - Navigate to the version that suits you and complete the installation using the wizard.
  
# Installing  
Required python libraries:
  - pandas - `pip install pandas`
  - sklearn - `pip install sklearn`
  - joblib - `pip install joblib`
  - tkinter - This should already be included in your Python install
  - keras - https://www.tutorialspoint.com/keras/keras_installation.htm
  - tensorflow - https://www.tensorflow.org/install/pip
  
After installation, run the train_model you would like to test with. Check the output name of the model `model.save('vehicle_value_model.h5)`. Navigate to use_model.py and change the input for line 9 (`model = joblib.load(<your train model here>)`) to the model that you would like to test against. 

# Running the tests
Run the use_model and the UI will load with several input forms, input the Year, Make, Model, Condition, Title Status, Transmission, and Odometer reading of the vehicle you would like to test.
- Year: any integer value (1950 - 2019 for best results)
- Make: Most common makes are included int the training data, if you are unsure open the vehicles.csv and see if that make is included.
- Model: Most common models are included in the training data, if you are unsure open the vehicles.csv and see if that model is included.
- Condtion: Ordinal value (fair, good, excellent, like new)
- Status: Categorical value (missing, rebuilt, salvage, clean)
- Transmission Type: manual/automatic
- Odometer: integer (in miles)

# Authors
 - Vincent Andrea
 
# License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details
