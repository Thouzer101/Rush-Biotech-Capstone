setwd('E:\\OneDrive - rush.edu\\Research Capstone\\Rush-Biotech-Capstone\\logistic_regression')

diabetes = read.csv('diabetes.csv', stringsAsFactors = TRUE)

logMod = glm(diabetes ~ age + bmi_class, family=binomial(), data=diabetes)

summary(logMod)