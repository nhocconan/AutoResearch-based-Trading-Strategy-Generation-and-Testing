#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + 12h EMA Trend Filter
# Hypothesis: Williams %R identifies overbought/oversold conditions; trade reversals in direction of 12h EMA trend.
# Works in bull/bear by fading extremes with trend filter. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_williams_r_12h_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R (14-period) on 6h
    wr_period = 14
    highest_high = pd.Series(high).rolling(window=wr_period, min_periods=wr_period).max().values
    lowest_low = pd.Series(low).rolling(window=wr_period, min_periods=wr_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold or trend changes
            if williams_r[i] > -20 or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought or trend changes
            if williams_r[i] < -80 or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: oversold with uptrend
            if williams_r[i] < -80 and close[i] > ema_50_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: overbought with downtrend
            elif williams_r[i] > -20 and close[i] < ema_50_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals