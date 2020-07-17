## Polynomial Regression
This project uses the same dataset as the Linear Regression project ( [IBM Object Storage](https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv) )
and creates a regression model to predict CO2 emissions. It does the process of regression by fitting, in this case, only one dataset feature (ENGINESIZE) while creating and replacing a polynomial equation, as explained below.

Although polynomial regression fits nonlinear relationships between the variables, it is still considered a linear regression
since it is linear in the regression coeficients - The equation's created features are replaced: x for 𝑥₁, x² for 𝑥₂ and so on.

Example of second degree polynomial equation: 𝑦 = 𝑏 + 𝜃₁𝑥 + 𝜃₂𝑥² 

Replaced equation: 𝑦 = 𝑏 + 𝜃₁𝑥₁ + 𝜃₂𝑥₂


![ ](./imgp/deg2.png)
![ ](./imgp/deg3.png)

PR - Degree 2 | PR - Degree 3
------------ | -------------
Mean absolute error: 23.24 | Mean absolute error: 23.10
Residual sum of squares (MSE): 850.71 | Residual sum of squares (MSE): 844.13
R2-score: 0.72 | R2-score: 0.72
