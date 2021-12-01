
from typing import (
    Union, Mapping, List,
)
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
from pypfopt.efficient_frontier import EfficientFrontier

class PortfOpt(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, mu, sigma):
        return NotImplementedError

class OptSharpeRatio(PortfOpt):
    def solve(self, mu, sigma):
        ef = EfficientFrontier(mu, sigma)
        ef.max_sharpe()
        weights = ef.clean_weights()
        return weights

class OptRisk(PortfOpt):
    def __init__(self, target_return: float = None):
        self.target_return = target_return

    def solve(self, mu, sigma, target_return: float = None):
        target_return = target_return or self.target_return
        ef = EfficientFrontier(mu, sigma)
        ef.efficient_return(target_return)
        weights = ef.clean_weights()
        return weights

class OptReturn(PortfOpt):
    def __init__(self, target_volatility: float = None):
        self.target_volatility = target_volatility

    def solve(self, mu, sigma, target_volatility: float = None):
        target_volatility = target_volatility or self.target_volatility
        ef = EfficientFrontier(mu, sigma)
        ef.efficient_risk(target_volatility)
        weights = ef.clean_weights()
        return weights



class MuFactory(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, rets: DataFrame):
        return NotImplementedError

class Average(MuFactory):
    def __call__(self, rets: DataFrame):
        return rets.mean()

class ExponentiallyMovingAverage(MuFactory):
    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.args = {
            'com': com,
            'span': span,
            'halflife': halflife,
            'alpha': alpha,
        }

    def __call__(self, rets: DataFrame, **kwds):
        tmp = self.args.copy()
        tmp.update(kwds)
        mu = rets.ewm(**tmp).mean().iloc[-1]
        return mu



class SigmaFactory(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, rets, stockembs):
        return NotImplementedError

class Covariance(SigmaFactory):
    def __call__(self, rets: DataFrame = None, stockembs: DataFrame = None):
        assert rets is not None
        return rets.cov()

class CovarianceShrinkage(SigmaFactory):
    def __init__(self, shrink=0):
        self.shrink = shrink
    
    def __call__(self, rets: DataFrame = None, stockembs: DataFrame = None, shrink: float = None):
        assert rets is not None
        shrink = shrink or self.shrink
        n = rets.shape[1]       # num of stocks

        cov = rets.cov()
        trace = np.trace(cov.values)
        sigma = cov * (1 - shrink) + np.eye(n) * trace / n * shrink
        return sigma

class StockembDotProduct(SigmaFactory):
    def __call__(self, rets: DataFrame = None, stockembs: DataFrame = None):
        assert stockembs is not None
        sigma = stockembs.values.T @ stockembs.values
        return sigma

class StockembCosine(SigmaFactory):
    def __call__(self, rets: DataFrame = None, stockembs: DataFrame = None):
        assert stockembs is not None
        norm = (stockembs.values ** 2).sum(axis=0, keepdims=True) ** 0.5   # (1, n)
        sigma = stockembs.values.T @ stockembs.values / (norm.T * norm)
        return sigma

class CovarianceTimesStockembDotProduct(SigmaFactory):
    def __call__(self, rets: DataFrame = None, stockembs: DataFrame = None):
        assert rets is not None and stockembs is not None
        cov = rets.cov()
        dotprod = stockembs.values.T @ stockembs.values
        return cov * dotprod




class Portfolio:
    timestamps    : pd.DatetimeIndex
    weights       : Mapping[str, float]  # weight of every stock
    returns       : Series               # portfolio's return per timestep

    def __init__(self, timestamps, weight, returns=None):
        self.timestamps   = timestamps
        self.weights      = weight
        self.returns      = returns
    
    def compute_portfolio_return(self, rets):
        returns = (rets.loc[self.timestamps] * self.weights).sum(axis=1)
        return returns
    
    def compute_portfolio_return_(self, rets):
        self.returns = self.compute_portfolio_return(rets)

    @property
    def stocks(self):
        return list(self.weights.index)
    
    @property
    def overall_return(self):
        return self.returns.sum()
    
    def __repr__(self):
        sorted_stock2weight = sorted(self.weights.items(), key=lambda x:x[1], reverse=True)
        s = ', '.join([f'{stock}={weight}' for stock, weight in sorted_stock2weight if weight > 0])
        return f'<Portfolio({s})>'
    

class Simulator:

    opt: PortfOpt
    mu_factory: MuFactory
    sigma_factory: SigmaFactory

    def __init__(self,
                 mu_factory       : Union[str, MuFactory],
                 sigma_factory    : Union[str, SigmaFactory],
                 opt              : Union[str, PortfOpt],
                 num_workers=4):
        if isinstance(mu_factory, str):
            mu_factory = {
                'Average'                       : Average,
                'ExponentiallyMovingAverage'    : ExponentiallyMovingAverage,
            }[mu_factory]()
        self.mu_factory = mu_factory

        if isinstance(sigma_factory, str):
            sigma_factory = {
                'Covariance'                          : Covariance,
                'CovarainceShrinkage'                 : CovarianceShrinkage,
                'StockembDotProduct'                  : StockembDotProduct,
                'StockembCosine'                      : StockembCosine,
                'CovarianceTimesStockembDotProduct'   : CovarianceTimesStockembDotProduct,
            }[sigma_factory]()
        self.sigma_factory = sigma_factory

        if isinstance(opt, str):
            opt = {
                'OptSharpeRatio'    : OptSharpeRatio,
                'OptRisk'           : OptRisk,
                'OptReturn'         : OptReturn,
            }[opt]()
        self.opt = opt

        self.num_workers = num_workers
    
    def simulate(self, rets: DataFrame, stockembs: DataFrame = None,
                 mukwds: dict = {}, sigmakwds: dict = {}, optkwds: dict = {}):
        '''
        Returns:
            portfolio: Portfolio
        '''
        mu        = self.mu_factory(rets, **mukwds)
        sigma     = self.sigma_factory(rets, stockembs, **sigmakwds)
        weights   = self.opt.solve(mu, sigma, **optkwds)
        portfolio = Portfolio(rets.index, weights)
        portfolio.compute_portfolio_return_(rets)
        return portfolio

    def simulate_multistep(self, rets: DataFrame, time2stockembs: Mapping[str, DataFrame] = None, 
                           mukwds: dict = {}, sigmakwds: dict = {}, optkwds: dict = {}):
        '''
        Returns:
            portfolios: List[Portfolio]
        '''

        ts = sorted(list(time2stockembs.keys()))
        def split_returns():
            tindex = rets.index
            retsegs = []
            for tstart, tend in zip(ts, ts[1:]+[None]):
                stocks = time2stockembs[tstart].columns
                retseg = rets[stocks].loc[tindex > tstart].loc[:tend]
                retsegs.append(retseg)
            return retsegs
        retsegs = split_returns()

        portfolios = []
        for retseg, t in zip(retsegs, ts):
            stockembs = time2stockembs[t]
            portfolio = self.simulate(retseg, stockembs, mukwds=mukwds, sigmakwds=sigmakwds, optkwds=optkwds)
            portfolios.append(portfolio)

        return portfolios