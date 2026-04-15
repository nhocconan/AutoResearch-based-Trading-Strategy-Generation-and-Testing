#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d Trend Filter and Volume Spike
# Williams %R identifies overbought/oversold conditions. Combined with 1d EMA trend filter
# and volume spikes, it captures mean reversion in trending markets.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 12h
    # We need 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low + 1e-10) * -100
    
    # Align Williams %R to 12h timeframe (already on 12h, but need to align to 12h index of prices)
    # Since we're using 12h data, we need to align it to the 12h resolution in prices
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume spike detection: volume > 2.0 * 20-period median
    volume_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    volume_spike = volume > 2.0 * volume_median
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) + price above 1d EMA (uptrend) + volume spike
        if (williams_r_aligned[i] < -80 and
            close[i] > ema_34_1d_aligned[i] and
            volume_spike[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + price below 1d EMA (downtrend) + volume spike
        elif (williams_r_aligned[i] > -20 and
              close[i] < ema_34_1d_aligned[i] and
              volume_spike[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R crosses back through -50 (mean reversion) or opposite signal
        elif position == 1 and (williams_r_aligned[i] > -50 or 
                               (williams_r_aligned[i] > -20 and close[i] < ema_34_1d_aligned[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r_aligned[i] < -50 or 
                                (williams_r_aligned[i] < -80 and close[i] > ema_34_1d_aligned[i])):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_Trend_Volume"
timeframe = "12h"
leverage = 1.0