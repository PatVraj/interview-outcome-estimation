# Speech-Based Machine Learning for Job Interview Outcome Prediction  

## Overview  
This project aims to design explainable machine learning models for predicting job interview outcomes, such as interviewee performance and excitement levels, using speech data. The models leverage both language and prosodic features to provide actionable feedback and ensure interpretability.  

The study uses the **MIT Interview dataset**, which includes transcripts, prosodic features, and scores for interview performance. A key focus is on explainable AI (XAI) techniques, ensuring that the model's decision-making process is transparent and understandable.  

---

## Features  
### Key Components of the Project  
1. **Language Feature Extraction**  
   - Extract syntactic features (e.g., TF-IDF, count vectorizer).  
   - Extract semantic features (e.g., sentiment analysis, topic modeling, word embeddings).  

2. **Language Feature Selection**  
   - Apply filter-based methods to identify relevant and interpretable features.  

3. **Interview Outcome Estimation**  
   - Use tree-based and deep learning models to estimate outcomes.  
   - Evaluate using Pearson’s correlation and absolute relative error (RE).  

4. **Multimodal Models**  
   - Combine language and prosodic features for multimodal learning.  
   - Compare performance of unimodal vs. multimodal approaches.  

5. **Explainable AI (XAI)**  
   - Implement techniques like SHAP, LIME, and EBM to interpret model decisions.  

6. **Transformer-based Experimentation (Bonus)**  
   - Explore transformer models (e.g., minGPT) for outcome prediction and explanation generation.  

---

## Dataset  
The **MIT Interview dataset** includes:  
- `transcripts.csv`: Transcripts of interviews with participant IDs.  
- `prosodic_features.csv`: Averaged prosodic features for each interview.  
- `scores.csv`: Ratings of interviewee performance and excitement (1-7 scale).  

---

## Methodology  
1. **Preprocessing**  
   - Clean and prepare transcript text.  
   - Normalize and aggregate prosodic features.  

2. **Feature Engineering**  
   - Extract syntactic and semantic language features.  
   - Identify top features using filter-based methods.  

3. **Model Training and Evaluation**  
   - Train tree-based and deep learning models on language features.  
   - Train multimodal models combining language and prosodic features.  
   - Evaluate models using correlation (r) and relative error (RE).  

4. **Explainability**  
   - Apply SHAP and LIME to interpret feature contributions.  
   - Compare explainability metrics across models.  

---

## Tools and Libraries  
- **Data Processing:** pandas, numpy, NLTK, scikit-learn  
- **Modeling:** scikit-learn, TensorFlow, PyTorch  
- **Feature Extraction:** TF-IDF, CountVectorizer, VADER  
- **Explainability:** SHAP, LIME, EBM  

---

## Results  
- **Unimodal Models:** Performance of language-based and prosodic-based models.  
- **Multimodal Models:** Improvement in prediction accuracy by combining modalities.  
- **Explainability Analysis:** Insights from SHAP and LIME visualizations.  

---

## How to Use  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/speech-interview-outcome-prediction.git  
   cd speech-interview-outcome-prediction  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the pipeline:  
   - Preprocessing: `python preprocess.py`  
   - Model training: `python train_model.py`  
   - Explainability analysis: `python explain.py`  

---

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

---

## Acknowledgments  
- **MIT Interview Dataset**  
- Tools like SHAP, LIME, and VADER for enabling this research.  

Feel free to contribute or provide feedback!  

---
