import pandas as pd
import numpy as np

# Creating a synthetic dataset
np.random.seed(0)
data = pd.DataFrame({
    'bedrooms': np.random.randint(1, 6, 100),
    'square_footage': np.random.randint(500, 3500, 100),
    'price': np.random.randint(100000, 500000, 100)
})

data.to_csv('house_prices.csv', index=False)