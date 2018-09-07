from datacl import Clean
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

clean=Clean()
df=clean.load_excel("D:/Github/BigData-project/all0830.xlsx")

y=df.iloc[:, :2]
#print(y)

model=smf.ols("price ~ temp", data=y)
res=model.fit()

#print(df.corr(method='pearson'))
#print(res.summary())
#print(res.params)

#fig = sm.graphics.plot_regress_exog(res, "temp")



#df.plot.boxplot() #표준화 필요

for i in range(1, len(df.ix[1,:])):
    plt.boxplot(df.ix[:,i][df.ix[:,i].notnull()])
    plt.show()
    
#df.ix[:,:1].boxplot()
#df.ix[:, 2:3].boxplot()
df.ix[:, 6:7].boxplot()