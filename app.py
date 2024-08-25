from flask import Flask, render_template, request, jsonify
import joblib
import cv2
import os

app = Flask(__name__, static_url_path='/static')

# Load the models
fertilizer_model = joblib.load('fertilizer_recommendation_model.pkl')
crop_model = joblib.load('crop_recommendation_model.pkl')
disease_model = joblib.load('trained_model.pkl')
pca_model = joblib.load('pca_model.pkl')

# Define class labels for disease prediction
class_labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Define a function to preprocess the uploaded image for disease prediction
def preprocess_image(img):
    img = cv2.resize(img, (200, 200))
    img = img.reshape(1, -1) / 255.0
    img = pca_model.transform(img)  # Apply PCA transformation
    return img

# Define a function for preprocessing input data for crop recommendation
def preprocess_crop_input(N, P, K, temperature, humidity, ph, rainfall):
    return [N, P, K, temperature, humidity, ph, rainfall]

# Define a function for preprocessing input data for fertilizer recommendation
def preprocess_fertilizer_input(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type):
    # Encode Soil Type and Crop Type as dummy variables
    soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Barley', 'Millets', 'Oil seeds', 'Pulses', 'Paddy', 'Ground Nuts', 'Wheat']
    
    soil_encoded = [1 if soil == soil_type else 0 for soil in soil_types]
    crop_encoded = [1 if crop == crop_type else 0 for crop in crop_types]
    
    return [temperature, humidity, moisture, nitrogen, potassium, phosphorous] + soil_encoded + crop_encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    fertilizer_procedures = {
        '17-17-17': 'Apply 50-100 kg/ha during land preparation. Repeat the application during crop growth stages based on soil and crop requirements.',
        '20-20': 'Apply 75-150 kg/ha before sowing or transplanting. Apply another dose during the vegetative growth stage if necessary.',
        '28-28': 'Apply 100-200 kg/ha as a basal dose during land preparation. Top-dress with 50 kg/ha during the vegetative stage.',
        'Urea': 'Apply 50-100 kg/ha during land preparation. Split the remaining dose into 2-3 applications during the crop growth stage.',
        '14-35-14': 'Apply 25-50 kg/ha during sowing or transplanting. Apply another dose during the flowering stage if required.',
        '10-26-26': 'Apply 25-50 kg/ha before sowing or transplanting. Apply additional doses during the flowering and fruiting stages as needed.',
        'DAP': 'Apply 50-100 kg/ha during land preparation. Repeat the application during crop growth stages based on soil and crop requirements.'
    }

    if request.method == 'POST':
        # Process form data and make fertilizer recommendation
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        nitrogen = float(request.form['nitrogen'])
        potassium = float(request.form['potassium'])
        phosphorous = float(request.form['phosphorous'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']

        # Perform necessary preprocessing and feature encoding
        processed_data = preprocess_fertilizer_input(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type)

        # Make prediction using the model
        recommendation = fertilizer_model.predict([processed_data])[0]

        # Get the procedure for the recommended fertilizer
        procedure = fertilizer_procedures.get(recommendation, 'No specific procedure available.')

        return jsonify({'recommendation': recommendation, 'procedure': procedure})
    else:
        return render_template('fertilizer_recommendation.html')

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        try:
            # Process form data and make crop recommendation
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Perform necessary preprocessing
            processed_data = preprocess_crop_input(N, P, K, temperature, humidity, ph, rainfall)

            # Make prediction using the model
            recommendation = crop_model.predict([processed_data])[0]

            crop_recommendations = {
                'Maize': 'Plant maize seeds at a depth of 2-3 cm in well-drained soil. Maintain soil moisture and provide fertilizer rich in nitrogen.',
                'Cotton': 'Plant cotton seeds in fertile, well-drained soil. Water regularly and provide balanced fertilizer. Monitor for pests and diseases.',
                'Chickpea': 'Sow chickpea seeds in well-drained soil. Provide moderate water and fertilize with phosphorus and potassium. Control weeds and pests as needed.',
                # Add recommendations for other crops here
            }

            # Get the procedure to grow for the recommended crop
            procedure_to_grow = crop_recommendations.get(recommendation)

            # Debug prints
            print("Recommended crop:", recommendation)
            print("Available crop recommendations:", crop_recommendations.keys())

            return render_template('crop_recommendation.html', recommendation=recommendation, procedure_to_grow=procedure_to_grow)
        
        except Exception as e:
            print("Error occurred:", str(e))
            return render_template('crop_recommendation.html', error_message="An error occurred while processing your request.")
    else:
        return render_template('crop_recommendation.html')


@app.route('/disease_prediction', methods=['GET', 'POST'])
def disease_prediction():
    prediction = None
    prevention_recommendation = None
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return render_template('disease_prediction.html', prediction="No file uploaded")
        
        file = request.files['image']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('disease_prediction.html', prediction="No file selected")
        
        # Save the uploaded image to a temporary location
        img_path = os.path.join(app.root_path, 'static', 'uploaded_image.jpg')
        file.save(img_path)

        # Read the uploaded image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image
        img = preprocess_image(img)

        # Perform prediction
        prediction_label = disease_model.predict(img)[0]
        prediction = class_labels[prediction_label]

        # Determine prevention recommendations based on predicted disease
        if prediction == 'Healthy':
            prevention_recommendation = "Keep your plants well-nourished and watered. Ensure proper sunlight and soil drainage."
        elif prediction == 'Powdery':
            prevention_recommendation = "Avoid overhead irrigation. Prune infected leaves. Apply fungicides if necessary."
        elif prediction == 'Rust':
            prevention_recommendation = "Ensure good air circulation around plants. Remove and destroy infected leaves. Apply fungicides as needed."
        else:
            prevention_recommendation = "No specific prevention recommendation available."

    return render_template('disease_prediction.html', prediction=prediction, prevention_recommendation=prevention_recommendation)

if __name__ == '__main__':
    app.run(debug=True)