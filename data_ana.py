from datacl import Clean
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

clean=Clean()
df=clean.load_excel("D:/Github/BigData-project/all0830.xlsx")

y=df.iloc[:, :2]
print(y)

model=smf.ols("price ~ temp", data=y)
res=model.fit()

print(df.corr(method='pearson'))
print(res.summary())
print(res.params)

fig = sm.graphics.plot_regress_exog(res, "temp")