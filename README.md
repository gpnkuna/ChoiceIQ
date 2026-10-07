# ChoiceIQ - IT Career Recommendation System

**Course:** BICT242: Data Scalability and Analytics  
**Project:** Developing a Recommendation System for an Application of Your Choice

ChoiceIQ is a student-facing IT career recommendation system. A student answers a 10-question interest and work-preference quiz, the application converts those answers into a nine-feature profile, and a content-based cosine k-nearest-neighbour model recommends the three most similar IT careers.

## Quick start

From this folder, run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens locally in the browser. The student completes all 10 questions and selects **Find my careers** to receive the top three recommendations.


## Data and feature engineering

The project uses ESCO occupation, skill, occupation-skill relationship, ISCO group and digital-skills files. The final recommendation dataset contains 30 selected ICT careers and 2,221 retained career-skill relationships.

Career profiles are represented by nine normalized features:

1. Data and analytics
2. Programming
3. Mathematics
4. Cybersecurity
5. Networking
6. Problem solving
7. Creativity and design
8. Systems and cloud
9. Communication and business

South African validation is provided separately using the supplied **2024 National List of Occupations in High Demand** and **2023 Critical Skills List**. Validation does not increase or decrease the recommendation similarity score.

## Recommendation model

Two content-based nearest-neighbour approaches were compared:

- Cosine k-NN
- Euclidean k-NN

The deployed model is cosine k-NN. It compares the student's nine-feature profile with the career feature matrix and returns the nearest three careers.

Current prototype benchmark results:

| Model | Precision@3 | Recall@3 | F1@3 | MRR |
|---|---:|---:|---:|---:|
| Cosine | 0.600 | 0.600 | 0.600 | 0.820 |
| Euclidean | 0.467 | 0.467 | 0.467 | 0.808 |

These metrics use 10 prototype benchmark profiles because historical real-student preference/rating data is not yet available. They should be interpreted as development evidence rather than population-level accuracy.



## Submission deliverables

This final package contains all four project deliverables:

1. **Data Analysis Report** - `reports/`
2. **Model Selection and Evaluation Report** - `reports/`
3. **Functional Recommendation System** - `app.py`
4. **Final Presentation** - `presentation/`

