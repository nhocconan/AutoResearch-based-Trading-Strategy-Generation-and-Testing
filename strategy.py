#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Combined with 1d EMA trend filter
# and volume spike to confirm momentum. Works in both bull and bear by only taking
# counter-trend moves during pullbacks in the direction of the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 12h
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low + 1e-10)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Williams %R crosses above -80 (oversold) + volume spike + price above 1d EMA50
        if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R crosses below -20 (overbought) + volume spike + price below 1d EMA50
        elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Williams %R signal
        elif position == 1 and williams_r_aligned[i] < -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and williams_r_aligned[i] > -50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_Trend_Filter"
timeframe = "12h"
leverage = 1.0