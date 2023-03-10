library('pROC')

pred = runif(100)

true = (runif(100) > 0.5)
true[true == TRUE] = 1

r = roc(true, pred)
print(r)

plot.roc(r, legacy.axes=TRUE, 
            print.auc=TRUE, 
            auc.polygon=TRUE,
            auc.polygon.col=rgb(0.0,0.0,0.7,0.2),
            )

