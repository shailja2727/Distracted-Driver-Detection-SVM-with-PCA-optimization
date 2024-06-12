
# Driver Distraction Detection System Based On SVM & PCA with Prioritized Alerting

The application features various user interfaces and output screens to facilitate image upload, prediction display, and alerting mechanisms. This section will discuss the key components and results of the application.

## User Interfaces and Output Screens

### 1.1 Home Page
The home page of the application welcomes the user and provides a way to navigate to the main functionality of the application. The output screen “home.html” can be seen in Fig. 1.1.

![image](https://github.com/shailja2727/Distracted-Driver-Detection-SVM-with-PCA-optimization/assets/99111449/bcc65f1a-01a0-4f9a-bf71-3d180ed6b6d2)

                     Fig. 1.1 Home page Screenshot


### 1.2 Main Page
The main page prompts the user to upload the image for prediction as shown in Fig. 1.2. The interface includes an input field to upload an image file, a button to submit the uploaded image for prediction, a section to display the prediction results and the uploaded image.

![image](https://github.com/shailja2727/Distracted-Driver-Detection-SVM-with-PCA-optimization/assets/99111449/09a57ba1-7702-468a-baee-26c245b4e957)

                        Fig. 1.2 Main page Screenshot

The main window displayed a "Select Image" button and a result label. Clicking the "Select Image" button opened a file dialog that allowed us to choose an image file from any directory as shown in Fig. 1.3.
 
 ![image](https://github.com/shailja2727/Distracted-Driver-Detection-SVM-with-PCA-optimization/assets/99111449/2d5c1373-2e8d-4077-a1bb-316701488b00)

                            Fig. 1.3 File Dialog
### 1.3 Prediction Output
After uploading an image, the application processes the image and displays the predicted distraction category along with the uploaded image. If the model detects a distraction, it triggers an alert based on the risk level associated with the detected category.
There are 2 options in the application; one is to upload the image of the driver that is to be predicted and the other one is to ‘predict’ the output and then start the warning with the alarm. The system first calls out the warning and then the alarm and then displays the uploaded image on the screen as shown in Fig.1.4.

 ![image](https://github.com/shailja2727/Distracted-Driver-Detection-SVM-with-PCA-optimization/assets/99111449/c55445dc-3aa6-4e09-8ec0-75c7c754ae5c)

                        Fig. 1.4 Prediction Output

When the system automatically identifies the fatality level and plays the warning with the alarm, the “app.py” logs the identified risk and alerting latency using the print statements. Fig. 1.5 shows the output of the print statements.
![image](https://github.com/shailja2727/Distracted-Driver-Detection-SVM-with-PCA-optimization/assets/99111449/a07b82ef-8e61-447c-ab7b-8d8cceb5a3e4)

        Fig. 1.5 Fatality Level Identified and Alerting Latency


