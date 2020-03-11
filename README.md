# Breast_Cancer_Reseach
Predicting the category of tumor using provided variables based on cell nucleus measurement.

An accurate prediction of a tumor is of upmost importance for hospitals and patients.  This project determines which, if any, measurements of a cancer cell nuclei contribute to an accurate diagnosis of an existing breast tumor.  The two classifications of tumor in this case are malignant and benign.  This project will be using data provided by the following study:

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

Using the provided data set, we will determine which if any of the independent variables are reliable in predicting the classification of a breast tumor.  Because of the severity of a type II error in the diagnosis of a tumor, considerations will be taken that may limit the acceptable models.

The ten independent variables that will be examined in this analysis are:

Ten real-valued features are computed for each cell nucleus:
a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The null hypothesis states that there is no correlation between the successful diagnosis of a breast tumor and measurements of a breast cancer cell nuclei that is minimal in type II error.  In order to reject the null hypothesis, there not only needs to be correlation, but correlation that will not lead to missed diagnosis of a malignant tumor.
