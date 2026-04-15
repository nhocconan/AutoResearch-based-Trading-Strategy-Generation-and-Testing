#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter
# Uses Williams %R(14) for overbought/oversold signals on 12h timeframe.
# Only takes longs when %R < -80 (oversold) and price > 1d EMA50 (uptrend).
# Only takes shorts when %R > -20 (overbought) and price < 1d EMA50 (downtrend).
# Includes volume confirmation to filter weak signals.
# Designed to work in both bull and bear markets by aligning with 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) on 12h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14 + 1e-10) * -100
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) + volume spike + price above 1d EMA50 (uptrend)
        if (williams_r_aligned[i] < -80 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + volume spike + price below 1d EMA50 (downtrend)
        elif (williams_r_aligned[i] > -20 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or Williams %R crosses -50 (mean reversion complete)
        elif position == 1 and williams_r_aligned[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and williams_r_aligned[i] < -50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_MeanReversion_TrendFilter"
timeframe = "12h"
leverage = 1.0