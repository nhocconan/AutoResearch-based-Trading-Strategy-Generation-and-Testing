#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme readings with volume confirmation and 4h EMA50 trend filter
# Long when 1d Williams %R < -80 (oversold) AND price > 4h EMA50 AND volume > 1.5 * avg_volume(20)
# Short when 1d Williams %R > -20 (overbought) AND price < 4h EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when Williams %R returns to -50 level or opposite extreme touched
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R identifies overextended moves likely to reverse
# Volume filter ensures institutional participation, reducing false signals
# EMA50 filter ensures trading with the intermediate-term trend
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

name = "4h_1dWilliamsR_Extreme_Volume_EMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 periods for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL) where HH=highest high, LL=lowest low over period
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 4h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 4h EMA50 for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d Williams %R oversold (< -80) AND price > 4h EMA50 AND volume confirmation
            if (williams_r_aligned[i] < -80 and close[i] > ema_50[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d Williams %R overbought (> -20) AND price < 4h EMA50 AND volume confirmation
            elif (williams_r_aligned[i] > -20 and close[i] < ema_50[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 or reaches overbought (> -20)
            if williams_r_aligned[i] >= -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 or reaches oversold (< -80)
            if williams_r_aligned[i] <= -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals