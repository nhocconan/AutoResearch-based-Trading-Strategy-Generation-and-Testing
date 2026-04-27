#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume-weighted Hurst Exponent + 1d Trend Filter
# Hurst Exponent measures trend persistence (H>0.5) vs mean reversion (H<0.5).
# Combines with 1d EMA trend filter and volume confirmation to capture strong trends.
# Works in bull (trending up) and bear (trending down) by following the 1d trend.
# Target: 15-25 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 12-period Hurst Exponent on 12h data
    def hurst_exponent(price_series, lags=12):
        """Calculate Hurst Exponent using R/S analysis"""
        n = len(price_series)
        if n < lags * 2:
            return np.full(n, np.nan)
        
        hurst_vals = np.full(n, np.nan)
        for i in range(lags * 2, n):
            # Get the last 'lags' observations
            ts = price_series[i-lags:i]
            if len(ts) < lags:
                continue
                
            # Calculate returns
            returns = np.diff(np.log(ts))
            if len(returns) == 0:
                continue
                
            # Mean of returns
            mu = np.mean(returns)
            
            # Cumulative deviations from mean
            cum_dev = np.cumsum(returns - mu)
            
            # Range (max - min)
            R = np.max(cum_dev) - np.min(cum_dev)
            
            # Standard deviation
            S = np.std(returns)
            
            # Avoid division by zero
            if S == 0:
                hurst_vals[i] = 0.5
            else:
                # R/S ratio
                RS = R / S
                # Hurst exponent approximation
                hurst_vals[i] = np.log(RS) / np.log(lags)
        
        return hurst_vals
    
    # Calculate Hurst Exponent
    hurst = hurst_exponent(close, lags=12)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(hurst[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: H > 0.5 (trending) AND price above EMA34 (uptrend) + volume
        if (hurst[i] > 0.5 and 
            close[i] > ema34_1d_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: H > 0.5 (trending) AND price below EMA34 (downtrend) + volume
        elif (hurst[i] > 0.5 and 
              close[i] < ema34_1d_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_HurstExponent_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0