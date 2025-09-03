## Email Spam Classification

## Project Overview
This project is an Email Spam Classifier built using Natural Language Processing (NLP) techniques and multiple machine learning algorithms. It aims to accurately classify incoming email messages into "Spam" or "Not Spam" (Ham) categories based on their content. The project involves data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment using a web app interface.

## Features
- Data cleaning and preprocessing including tokenization, stop word removal, and stemming.
- Visualization through plots and word clouds to explore spam vs ham email characteristics.
- Multiple classification algorithms including Naive Bayes variants, SVM, Logistic Regression, Random Forest, Gradient Boosting, and more.
- Evaluation using accuracy and precision metrics.
- Deployment-ready Streamlit web app for real-time spam detection.

## Repository Structure
```bash
email-spam-classifier/
│
├── data/
│ └── raw/
│ └── email_spam_detect_dataset.csv # Raw email dataset
│
├── notebooks/
│ └── EDA_and_Modeling.ipynb # Jupyter notebook containing EDA and model experiments
│
├── src/
│ ├── data_loader.py # Data loading functions
│ ├── preprocessing.py # Text preprocessing functions
│ ├── model.py # Model training, evaluation, and serialization
│ └── app.py # Streamlit app code for user interface and prediction
|── README.md
```


## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd email-spam-classifier
```

2. Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

Alternatively, install packages manually such as:
```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud streamlit xgboost
```

3. Download necessary NLTK resources:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```


## Usage

### Running the notebook
Open `notebooks/EDA_and_Modeling.ipynb` to explore data insights and train models interactively.

### Training and saving model
Use the functions in `src/model.py` to train your model and save the model and vectorizer for deployment.

### Launching the web app
Run the Streamlit application to classify emails in real time:
```bash
streamlit run src/app.py
```


Enter an email message in the text area and click "Predict" to see if it is spam or not.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check issues page and submit pull requests.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- Dataset source: [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
- Inspiration and methodology references from multiple machine learning tutorials and blog posts.
- Special thanks to the open-source community for tools like scikit-learn, NLTK, and Streamlit.

## Contact
For any inquiries or collaborations, please reach out via;
- Email : itsgazanfar@gmail.com
 - WhatsApp : +923053839897
- LinkedIn : www.linkedin.com/in/gazanfar-ali

