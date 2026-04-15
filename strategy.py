#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with Trend Filter and Volume Spike
# Williams %R identifies overbought/oversold conditions. Combined with EMA trend filter (50 EMA)
# and volume spike confirmation, it captures mean-reversion entries in trending markets.
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
# Target: 60-120 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Align Williams %R to 4h timeframe
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr)
    
    # Load 4h data for EMA50 trend filter
    ema_period = 50
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(willr_aligned[i]):
            continue
        
        # Long entry: Williams %R oversold (< -80) + price above EMA50 + volume spike
        if (willr_aligned[i] < -80 and
            close[i] > ema[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + price below EMA50 + volume spike
        elif (willr_aligned[i] > -20 and
              close[i] < ema[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R reverts to midpoint (-50) or opposite extreme
        elif position == 1 and (willr_aligned[i] > -50 or willr_aligned[i] > -20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (willr_aligned[i] < -50 or willr_aligned[i] < -80):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_Trend_Volume"
timeframe = "4h"
leverage = 1.0