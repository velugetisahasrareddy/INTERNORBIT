# INTERNORBIT
Movie Rating Prediction using Python  Predicts IMDb movie ratings based on genre, director, and actors using regression models (Linear Regression, Random Forest, Gradient Boosting). Built with the IMDb India dataset. Best model: Gradient Boosting with R2 = 0.334 and MAE = 0.851.
Here's a clean README you can copy directly:

```markdown
# Movie Rating Prediction with Python

A machine learning project that predicts IMDb movie ratings based on 
genre, director, and actors.

## Dataset
IMDb India Movies Dataset (15,509 movies)

## Models Used
- Linear Regression
- Random Forest
- Gradient Boosting

## Results
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 1.023 | 1.320 | 0.064 |
| Random Forest | 0.860 | 1.140 | 0.302 |
| Gradient Boosting | 0.851 | 1.113 | 0.334 |

Gradient Boosting performed best with R2 = 0.334 and MAE = 0.851.

## Features
- Year, Duration, Genre, Director, Actor 1, Actor 2, Actor 3, Votes

## Libraries
- pandas, numpy, scikit-learn, matplotlib

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
python movie_rating_prediction.py
```

## Internship
Built as part of the Intern Orbit Data Science Internship - Level 1 Task 1.
```

