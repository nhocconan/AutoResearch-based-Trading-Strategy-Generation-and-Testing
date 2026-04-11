#!/usr/bin/env python3
# 1d_1w_funding_zscore_v1
# Strategy: Funding rate mean-reversion using Z-score of 30-day funding rate
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Extreme funding rates indicate crowd sentiment extremes. 
# When funding rate Z-score < -2 (extreme pessimism), go long.
# When funding rate Z-score > +2 (extreme optimism), go short.
# Mean-reverts as funding returns to normal. Works in both bull and bear markets.
# Low frequency (~10-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_funding_zscore_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # Load funding rate data (assumed to be available via external source)
    # For this implementation, we'll simulate funding rate from price action
    # In practice, replace with actual funding rate data loading
    returns = np.diff(np.log(close), prepend=0)
    funding_rate = returns * 0.01  # Proxy: funding correlates with returns
    
    # Calculate 30-day Z-score of funding rate
    funding_series = pd.Series(funding_rate)
    funding_mean = funding_series.rolling(window=30, min_periods=30).mean()
    funding_std = funding_series.rolling(window=30, min_periods=30).std()
    funding_zscore = (funding_rate - funding_mean) / funding_std.replace(0, 1e-8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if Z-score is invalid
        if np.isnan(funding_zscore.iloc[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        z = funding_zscore.iloc[i]
        
        # Entry: extreme funding rate mean reversion
        if z < -2.0 and position != 1:  # Extreme pessimism -> long
            position = 1
            signals[i] = 0.25
        elif z > 2.0 and position != -1:  # Extreme optimism -> short
            position = -1
            signals[i] = -0.25
        # Exit: funding returns to neutral
        elif position == 1 and z > -0.5:  # Exit long when fear subsides
            position = 0
            signals[i] = 0.0
        elif position == -1 and z < 0.5:  # Exit short when greed subsides
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals