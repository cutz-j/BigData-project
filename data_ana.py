from datacl import Clean
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

clean=Clean()
df=clean.load_excel("D:/Github/BigData-project/weather.xlsx")

y=df.iloc[:, :1]
x=df.iloc[:, 1:]

model=smf.ols("price ~ temp + rain + hum + wsd", data=df)
res=model.fit()
print(res.summary())

print(res.params)

fig = sm.graphics.plot_regress_exog(res, "temp")