#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Mean Reversion with 1d Trend Filter
# Williams %R identifies overbought/oversold conditions. We buy when %R < -80 (oversold) 
# and sell when %R > -20 (overbought), but only in the direction of the 1d trend (EMA50).
# This combines mean reversion entries with trend filtering to work in both bull and bear markets.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) and price above 1d EMA50 (uptrend)
        if (williams_r[i] < -80 and
            close[i] > ema_50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) and price below 1d EMA50 (downtrend)
        elif (williams_r[i] > -20 and
              close[i] < ema_50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Williams %R signal or trend change
        elif position == 1 and (williams_r[i] > -20 or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r[i] < -80 or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_MeanReversion_TrendFilter"
timeframe = "12h"
leverage = 1.0