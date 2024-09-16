import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from st_functions import st_button, load_css
from PIL import Image
import streamlit_lottie 


def load_model(model_path):
    """Loads a saved machine learning model from a specified path.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        The loaded machine learning model, or None if loading fails.
    """

    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found at:", model_path)
        return None  # Indicate prediction failure
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None  # Indicate prediction failure

def predict_marks(hours_studied, model):
    """Predicts student marks based on hours studied using a loaded model.

    Args:
        hours_studied (float): The number of hours studied.
        model: The loaded machine learning model.

    Returns:
        float: The predicted student marks, or None if prediction fails.
    """

    try:
        # Prepare input data as a NumPy array (reshape if necessary)
        input_data = np.array([hours_studied]).reshape(1, -1)

        # Make predictions
        predicted_marks = model.predict(input_data)
        return predicted_marks[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None  # Indicate prediction failure

def home_page():
    st.header("Student Marks Prediction",divider=True)
    st.balloons()
    st.markdown('''
          :blue[WEL]:green[COME]: To predict:red[student marks] based on hours of study.  
          ''')

    
    prediction_page()
    st.markdown('''
          :blue[if you want to more details go to sidebar]:orange[| --> |] 
          ''')
   


       



  
        
     
def prediction_page():
    st.title("Prediction")

    hours_studied = st.number_input("Hours Studied", min_value=0.0, step=0.1)

    if st.button("Predict"):
        model = load_model("student_marks_predictor.pkl")
        if model:
            predicted_marks = predict_marks(hours_studied, model)
            if predicted_marks is not None:
                   st.write("Predicted Marks:", predicted_marks)
            else:
                st.error("Prediction failed. Please check model and input data.")
        else:
            st.error("Error loading model. Please check the model path.")

def about_page():
    st.title(" get in touch with me")
    st.markdown("""
    You :blue[can contact me] via social media or email. 
                """)
    st.balloons()
    
 
   
    load_css()

    
    col1, col2, col3 = st.columns(3)
    col2.image(Image.open("C:/Users/Manish/OneDrive/Desktop/circleimage.png"))

    st.header('manish kumar mishra,fullstack developer')

    st.info('fullstack developer, problem solver  with an interest in Machine Learning , Generative Ai')

    icon_size = 20

    st_button('github', 'https://github.com/manishmishra01', 'Follow me on github for source code', icon_size)
    st_button('medium', 'https://medium.com/@manishmishra9685', 'Follow me on Medium', icon_size)
    st_button('twitter', 'https://x.com/ManishX89?t=QnqabjiM1bEKDe7rnympuw&s=09', 'Follow me on Twitter', icon_size)
    st_button('linkedin', 'https://www.linkedin.com/in/manishmishra01', 'Follow me on LinkedIn', icon_size)
    st_button('Email', 'manishmihsra9685@gmail.com', 'Send me an Email(manishmishra9685@gmail.com)', icon_size)



def model_page():
    st.title("Model")
    st.header("The model is trained on...",divider="green")
    st.markdown(""" MODEl =  :red[student_marks_predictor.pkl] """)
    st.info("STEPS TO TRAIN THE MODEL: \n\n 1. Load the dataset. \n 2. Train the model. \n 3. Save the model.\n 4. Check accuracy of the model.")
    st.subheader("LIBARAY AND PACKAGES USED")
    st.code("import streamlit as st\nimport numpy as np\nimport joblib\nimport pickle\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression")
    st.text("""# here model selection = Linear Regression()
                
                model = LinearRegression()
               becouse dataset almost go to linear form that's why we use LinearRegression()
               
            

            """)
    st.latex(r"""y = mx+c""")


   

def dataset_page():
    st.title("Dataset")
    data = pd.read_csv("C:/Users/Manish/OneDrive/Desktop/student_info.csv")  # Replace with your dataset path

    # Display the dataset
    st.dataframe(data)

    # Download button
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download",
        data=csv,
        file_name="C:/Users/Manish/OneDrive/Desktop/student_info.csv",
        mime="text/csv",
    )
    st.text("Downloaded dataset as CSV file")

    st.subheader("cleaning dataset",divider="green")
    st.text("""before going to train the dataset we need to clean the dataset
               we can use pandas library to clean the dataset
               we use dropna() function to drop the null values
               check the null values using isnull().sum()
               fill the null values using fillna()
               store the new dtaframe in df2
               
               """)
    st.code(""" df.data.dropna(inplace=True)
                df.isnull().sum()
                df.mean()
                df = df.fillna(df.mean())
                df.isnull().sum()
                """)
    st.title("Student Hours vs. Marks")

    # Load the CSV data
    data = pd.read_csv("C:/Users/Manish/Downloads/student_info.csv")

    # Visualize the data
    visualize_data(data)
              
    
def visualize_data(data):
    """Visualizes the relationship between student hours and marks.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.
    """

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['study_hours'], data['student_marks'])
    plt.xlabel("Student Hours")
    plt.ylabel("Student Marks")
    plt.title("Relationship Between Student Hours and Marks")
    st.pyplot()

    # Perform linear regression
       # Perform linear regression
   


    # Predict marks
  

    # Plot the regression line
   
   



def main():
    st.sidebar.title("ML project")
    page = st.sidebar.selectbox("Details", ["Home", "Model","dataset", "About"])

    if page == "Home":
        home_page()
    elif page == "Model":
        model_page()
    elif page == "About":
        about_page()
    elif page == "dataset":
        dataset_page()
if __name__ == "__main__":
    main()