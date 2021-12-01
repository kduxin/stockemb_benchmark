
from abc import ABCMeta
import numpy as np
import pandas as pd
from pandas import Series
from .simulate import Portfolio
import powerlaw

class Score(metaclass=ABCMeta):
    pass

class Gain(Score):
    def __call__(self, portfolio: Portfolio, multiply=False):
        rets = portfolio.returns
        if multiply:
            score = (rets + 1).prod() - 1
        else:
            score = rets.sum()
        return score

class InformationRatio(Score):
    def __call__(self, portfolio: Portfolio, reference: Series,
                 annualize_factor: float = 252**0.5):
        rets = portfolio.returns
        excess_ret = rets - reference
        score = excess_ret.mean() / excess_ret.std()
        return score * annualize_factor

class SharpeRatio(Score):
    def __call__(self, portfolio: Portfolio, riskfree_rate: float,
                 annualize_factor: float = 252**0.5):
        rets = portfolio.returns
        score = (rets.mean() - riskfree_rate) / rets.std()
        return score * annualize_factor

class ParetoAlpha(Score):
    def __call__(self, portfolio, whichtail='negative'):
        assert whichtail in ['negative', 'positive', 'both']
        rets = portfolio.returns
        if whichtail == 'negative':
            rets = - rets.loc[rets < 0]
        elif whichtail == 'positive':
            rets = rets.loc[rets > 0]
        else:
            rets = rets.abs()
        
        fit = powerlaw.Fit(rets)
        score = fit.alpha - 1
        return score