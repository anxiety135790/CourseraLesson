import os.path

import matplotlib.pyplot as plt
import pandas as pd

sourceFile = "FuelConsumption.csv"

var = os.path.curdir

df = pd.read_csv(sourceFile)
df.head()
df.describe()

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz = cdf.head(9)

plt.scatter(viz.FUELCONSUMPTION_COMB, viz.CYLINDERS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
