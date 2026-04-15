#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d volume confirmation and 1w EMA trend filter
# Designed for low trade frequency (target 15-30/year) with clear mean-reversion logic
# Works in both bull (sell at overbought in uptrend) and bear (buy at oversold in downtrend) markets
# Uses Williams %R from daily, volume spike to confirm interest, and weekly EMA for trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R (14-period) from previous day to avoid look-ahead
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # Use previous day's Williams %R to avoid look-ahead
    williams_r_prev = np.concatenate([[np.nan], williams_r[:-1]])
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_prev)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size as fraction of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) + downtrend + volume spike
        if (williams_r_aligned[i] < -80 and 
            close[i] < ema50_1w_aligned[i] and 
            volume[i] > 1.8 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: Williams %R overbought (> -20) + uptrend + volume spike
        elif (williams_r_aligned[i] > -20 and 
              close[i] > ema50_1w_aligned[i] and 
              volume[i] > 1.8 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or Williams %R returns to neutral range (-50 to -50)
        elif position == 1 and williams_r_aligned[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and williams_r_aligned[i] < -50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_1dVolume_1wEMA_Reversal"
timeframe = "12h"
leverage = 1.0