library('pROC')

setwd('E:\\OneDrive - rush.edu\\Research Capstone\\Rush-Biotech-Capstone\\analysis')

csvData = read.csv('no_pretrain_results.csv')



r = roc(csvData$Label, csvData$Prediction)
print(r)

plot.roc(r, legacy.axes=TRUE, 
            print.auc=TRUE, 
            auc.polygon=TRUE,
            auc.polygon.col=rgb(0.0,0.0,0.7,0.2),
            )

