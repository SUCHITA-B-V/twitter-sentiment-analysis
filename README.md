# Twitter Sentiment Analysis Using Text Embeddings and Machine Learning

## 📌 Project Overview

Social media platforms generate a huge amount of text every day. Analyzing these posts manually to understand public opinion can be time-consuming.

This project develops a machine-learning-based sentiment classification system that analyzes tweets and classifies them into three categories: **Positive, Negative, and Neutral**.

The project uses a pretrained Sentence Transformer model to convert tweet text into numerical embeddings and a Logistic Regression classifier to predict sentiment.

## 🎯 Project Objective

The main objective is to build a sentiment classification pipeline that can:

- Analyze text from social media posts.
- Convert textual content into numerical representations using text embeddings.
- Classify tweets into positive, negative, or neutral sentiment categories.
- Evaluate model performance using standard classification metrics.
- Explore the dataset through visualizations and test predictions on custom tweets.

## 💡 Motivation

Understanding public opinion is valuable for businesses, governments, and organizations. For example, companies can analyze customer feedback to understand satisfaction levels and identify negative experiences.

Automated sentiment classification can help organize large volumes of social media text and provide an initial understanding of public opinion.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab / Jupyter Notebook | Development and execution environment |
| Pandas | Dataset loading and data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| WordCloud | Visualization of frequently occurring words |
| Sentence Transformers | Generating text embeddings |
| all-MiniLM-L6-v2 | Pretrained sentence-embedding model |
| Scikit-learn | Data splitting, classification, and evaluation |

## 📊 Dataset

**Dataset:** Twitter Tweets Sentiment Dataset

**Source:** [Kaggle – Twitter Tweets Sentiment Dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)

The dataset contains tweet text and corresponding sentiment labels.

### Dataset columns

| Column | Description |
|---|---|
| `text` | The text content of a tweet |
| `sentiment` | The sentiment label associated with the tweet |

### Sentiment categories

- **Positive:** Expresses satisfaction, happiness, approval, or other positive opinions.
- **Negative:** Expresses dissatisfaction, criticism, anger, or other negative opinions.
- **Neutral:** Expresses information or content without a clearly positive or negative opinion.

The notebook expects the dataset columns to be named `text` and `sentiment`. If the downloaded CSV uses different column names, update the notebook accordingly.

The dataset is not included in this repository. Download it from Kaggle and upload the CSV file when running the notebook.

## ⚙️ Project Workflow

The project follows these steps:

1. **Dataset Loading:** Load the labeled tweets into a Pandas DataFrame.
2. **Data Inspection:** Examine dataset dimensions, column names, missing values, and sentiment distribution.
3. **Text Preprocessing:** Remove rows with missing values and convert tweet text to lowercase.
4. **Exploratory Data Analysis:** Visualize sentiment distribution, frequent words, and text-length distribution.
5. **Embedding Generation:** Use the pretrained `all-MiniLM-L6-v2` model to convert each tweet into a numerical vector.
6. **Train-Test Split:** Divide the data into training and testing sets to evaluate performance on unseen examples.
7. **Model Training:** Train a Logistic Regression classifier using the training embeddings and their sentiment labels.
8. **Model Evaluation:** Measure classification performance using accuracy, precision, recall, F1-score, and a confusion matrix.
9. **Custom Predictions:** Test the trained classifier on new tweet examples.
10. **Cosine Similarity Analysis:** Compare tweet embeddings with average embeddings for each sentiment class as an additional similarity-based experiment.

*Note: The custom-prediction and cosine-similarity sections should be included in the notebook before describing them as completed features.*

## 🧠 Methodology

### 1. Text Preprocessing

The dataset is prepared for analysis by removing rows with missing values and converting text to lowercase.

This basic preprocessing step helps standardize text before embedding generation.

### 2. Text Embedding Generation

The project uses the pretrained Sentence Transformer model:

`all-MiniLM-L6-v2`

Text embeddings represent sentences as numerical vectors that capture useful semantic information.

The selected model produces a 384-dimensional embedding for each input sentence.

These embeddings are used as input features for the machine-learning classifier.

The embedding model is pretrained; it is not trained from scratch in this project.

### 3. Logistic Regression Classification

Logistic Regression is used as the supervised classification algorithm.

The classifier learns patterns from the training embeddings and their corresponding sentiment labels to predict whether a tweet is positive, negative, or neutral.

### 4. Cosine Similarity

Cosine similarity measures the similarity between numerical vectors.

As an additional experiment, a tweet embedding can be compared with the average training embedding of each sentiment category. The closest category can then be used as a similarity-based prediction.

This experiment is separate from Logistic Regression and can be used to compare two different prediction approaches.

## 📈 Exploratory Data Analysis

The notebook uses the following visualizations:

### Sentiment Distribution

A count plot displays the number of tweets in each sentiment category.

### Word Clouds

Separate word clouds help visualize frequently occurring words in tweets belonging to each sentiment category.

### Text-Length Distribution

A histogram displays the distribution of tweet lengths, measured by character count.

These visualizations help understand the dataset before evaluating the classification model.

## 📏 Model Evaluation

The model is evaluated on a held-out test set.

The following metrics are used:

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** The proportion of predictions for a particular class that are correct.
- **Recall:** The proportion of actual examples of a class that are correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion Matrix:** A table that compares actual labels with predicted labels.

### Model Performance

The following results should be filled in after executing the notebook.

| Metric | Result |
|---|---|
| Test accuracy | To be measured |
| Positive-class F1-score | To be measured |
| Negative-class F1-score | To be measured |
| Neutral-class F1-score | To be measured |

The actual classification report and confusion matrix generated by the executed notebook should be used to report final performance.

No model performance values are stated here because the final notebook results have not yet been verified.

## 🧪 Custom Tweet Predictions

The trained model can be tested using new tweet examples that were not used during training.

Example inputs include:

- "I absolutely love this product!"
- "This is a terrible experience."
- "The meeting starts at 10 AM tomorrow."
- "The service was excellent and the staff were friendly."
- "I am disappointed with the poor quality."

The notebook generates embeddings for these inputs and uses the trained classifier to predict their sentiments.

The actual predicted labels and estimated class probabilities should be obtained by executing the custom-prediction cell.

## 🚀 How to Run the Project

### Option 1: Run using Google Colab

1. Open the `sentiment_analysis.ipynb` notebook in this repository.
2. Open it in Google Colab.
3. Download the dataset from the Kaggle link provided above.
4. Extract the downloaded ZIP file and locate the CSV.
5. Upload the CSV when prompted by the notebook, or update the CSV path in the dataset-loading cell.
6. Ensure the dataset contains the required `text` and `sentiment` columns.
7. Execute the notebook cells in order.
8. Review the visualizations, model metrics, confusion matrix, and custom predictions.

### Option 2: Run locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/twitter-sentiment-analysis.git
```

Navigate to the project directory:

```bash
cd twitter-sentiment-analysis
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `sentiment_analysis.ipynb`, provide the dataset, and execute the cells in order.

The pretrained embedding model may need an internet connection for its initial download.

## 📦 Requirements

The main third-party packages used in the notebook are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
sentence-transformers
wordcloud
```
These packages can be installed using the requirements.txt file.

For reproducible results, package versions can be pinned after testing the project environment.

## ⚠️ Limitations

- Sarcasm and irony may lead to incorrect sentiment predictions.
- Slang, abbreviations, and context-dependent expressions can be difficult to interpret.
- Basic preprocessing does not address all text-cleaning challenges.
- Incorrect or inconsistent dataset labels may affect model performance.
- The classifier's predictions are estimates and may not represent the author's actual intention.
- Performance on the supplied dataset may differ from performance on tweets collected from other sources or time periods.

## 🔮 Future Improvements

Possible future improvements include:

- Improving text preprocessing and handling social media-specific language.
- Comparing Logistic Regression with additional classification algorithms.
- Performing hyperparameter tuning.
- Evaluating class-wise performance on different datasets.
- Exploring other embedding models.
- Developing an interactive interface using Streamlit.
- Testing the model on more diverse real-world text.

## 📚 Learning Outcomes

This project provides practical experience with:

- Text preprocessing and exploratory data analysis.
- Pretrained language embeddings.
- Supervised machine-learning classification.
- Training and testing data separation.
- Classification evaluation metrics.
- Confusion-matrix interpretation.
- Similarity-based analysis of text embeddings.

## 👩‍💻 Project Information

**Project Title:** Twitter Sentiment Analysis Using Text Embeddings and Machine Learning

**Domain:** Artificial Intelligence, Machine Learning, and Natural Language Processing

**Programming Language:** Python

**Development Environment:** Google Colab / Jupyter Notebook



