import functools
import matplotlib.pyplot as plt
import seaborn as sns
from base import Base


def show(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            plt.show()
    return wrapper


class Plot(Base):

    @show
    def mean_bar_by_target(self):
        # group by target since name is continuously
        mean = self.tdf[[self.NAME, self.TARGET]].groupby([self.TARGET], as_index=False).mean()
        sns.barplot(x=self.TARGET, y=self.NAME, data=mean)

    @show
    def mean_bar_by_name(self):
        # group by name since name is dispersed
        mean = self.tdf[[self.NAME, self.TARGET]].groupby([self.NAME], as_index=False).mean()
        sns.barplot(x=self.NAME, y=self.TARGET, data=mean)


class CPlot(Plot):
    HIST = 70

    @show
    def hist(self, bins=100):
        self.tdf[self.NAME].dropna().plot(kind='hist', bins=bins)

    @show
    def kde(self):
        facet = sns.FacetGrid(self.tdf, hue=self.TARGET, aspect=4)
        facet.map(sns.kdeplot, self.NAME, shade=True)
        facet.set(xlim=(0, self.tdf[self.NAME].max()))
        facet.add_legend()

    @show
    def mean_kde_by_name(self):
        sns.kdeplot(self.tdf[[self.NAME, self.TARGET]].groupby([self.NAME], as_index=False).mean())

    @show
    def mean_bar(self):
        # group by target since name is continuously
        self.mean_bar_by_target()

    def profile(self):
        self.hist(self.HIST)
        self.mean_bar()
        self.kde()
        self.mean_bar_by_name()
        self.mean_kde_by_name()


class DPlot(Plot):

    @show
    def count(self):
        _, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
        sns.countplot(x=self.NAME, data=self.tdf, ax=axis1)
        sns.countplot(x=self.TARGET, hue=self.NAME, data=self.tdf, ax=axis2)

    @show
    def mean_bar(self):
        # group by name since name is dispersed
        self.mean_bar_by_name()

    @show
    def point(self):
        sns.pointplot(self.NAME, self.TARGET, data=self.tdf, size=4, aspect=3)

    def profile(self):
        self.count()
        self.mean_bar()
        self.point()
