import numpy as np
from base import Base


class ETL(Base):

    def fill_na_by_mean(self):
        mean = self.tdf[self.NAME].mean()
        std = self.tdf[self.NAME].std()
        count_nan = self.tdf[self.NAME].isnull().sum()
        rand = np.random.randint(mean - std, mean + std, size = count_nan)
        self.tdf[self.NAME][np.isnan(self.tdf[self.NAME])] = rand
