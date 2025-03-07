# Resume-Score-
### Project Description: AI-Powered Resume Screening and Ranking System

Overview:
The AI-Powered Resume Screening and Ranking System** is an intelligent tool designed to automate and streamline the process of evaluating and ranking resumes based on their relevance to a specific job description. Leveraging Natural Language Processing (NLP) techniques, this system analyzes the content of resumes and compares them to the job description to identify the most suitable candidates. It eliminates manual effort, reduces bias, and ensures a more efficient hiring process.

This project is a **Resume Scoring and Ranking System** that uses natural language processing (NLP) and machine learning techniques to evaluate and rank resumes based on their relevance to a given job description. The system is designed to automate the process of resume screening by assigning a match score to each resume, indicating how well it aligns with the job requirements. The project is implemented using Python and integrates a Flask web application for user interaction.

### Key Components of the Project:

1. **Text Extraction**:
   - The system extracts text from PDF resumes using the `pdfminer` library. This allows it to process resumes submitted in PDF format.

2. **Text Cleaning**:
   - The extracted text is cleaned by removing special characters, converting it to lowercase, and eliminating unnecessary newlines. This step ensures that the text is in a consistent format for further processing.

3. **Dataset Loading**:
   - The project uses a sample dataset (`resume_dataset_path`) that contains resumes and their corresponding job categories. This dataset is used to train a machine learning model.

4. **Feature Extraction**:
   - The text data from the resumes is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization. This technique helps in representing the text data in a way that can be used by machine learning models.

5. **Model Training**:
   - A **Logistic Regression** model is trained on the dataset to classify resumes into different job categories. The model is evaluated using accuracy as the metric.

6. **Resume Ranking**:
   - The system ranks resumes based on their similarity to a given job description. This is done using **cosine similarity**, which measures the cosine of the angle between the TF-IDF vectors of the job description and the resumes. Higher similarity scores indicate a better match.

7. **Flask Web Application**:
   - The project includes a Flask web application that allows users to input a job description and a resume text. The system then calculates and displays the match score for the resume.

### Workflow:

1. **Text Extraction and Cleaning**:
   - The system extracts text from a resume PDF and cleans it for further processing.

2. **Training the Model**:
   - The system uses a dataset of resumes and their categories to train a logistic regression model. The model learns to classify resumes into different job categories based on the text content.

3. **Ranking Resumes**:
   - Given a job description, the system ranks all resumes in the dataset based on their similarity to the job description using cosine similarity.

4. **Web Interface**:
   - The Flask web app provides a user-friendly interface where users can input a job description and a resume text. The system calculates the match score and displays it to the user.

### Example Usage:

- A user uploads a resume and provides a job description.
- The system extracts and cleans the text from the resume.
- It then calculates the match score between the resume and the job description using cosine similarity.
- The match score is displayed to the user, indicating how well the resume matches the job description.

### Technologies Used:

- **Python**: The primary programming language used for the project.
- **pdfminer**: For extracting text from PDF resumes.
- **NLTK**: For text processing tasks such as stopword removal.
- **Pandas**: For handling and manipulating the dataset.
- **Scikit-learn**: For machine learning tasks, including TF-IDF vectorization, logistic regression, and cosine similarity.
- **Flask**: For building the web application.

### Potential Use Cases:

- **HR Departments**: Automating the initial screening of resumes for job applications.
- **Recruitment Agencies**: Quickly identifying the most relevant resumes for a given job posting.
- **Job Portals**: Providing resume ranking services to job seekers and employers.

### Limitations:

- The system relies on the quality of the dataset and the accuracy of the TF-IDF vectorization. If the dataset is small or not representative, the model's performance may be limited.
- The system assumes that the job description and resumes are in English. It may not perform well with resumes in other languages.
- The model's accuracy depends on the quality of the training data and the relevance of the features extracted.

Overall, this project demonstrates how machine learning and NLP can be used to automate and improve the resume screening process, saving time and effort for recruiters and HR professionals.

  


