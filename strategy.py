#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme reversal with 4h EMA34 trend filter and volume confirmation
# Long when 1d Williams %R crosses above -80 (oversold reversal) AND 4h EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when 1d Williams %R crosses below -20 (overbought reversal) AND 4h EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when 1d Williams %R crosses opposite extreme (-20 for long exit, -80 for short exit)
# Uses discrete sizing 0.25 to control drawdown and fees
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R provides mean-reversion signals at extremes, effective in both bull (buy dips) and bear (sell rallies)
# EMA34/EMA89 filter ensures trades align with intermediate trend to avoid counter-trend whipsaws
# High volume confirmation filters weak reversals and increases signal reliability

name = "4h_1dWilliamsR_Extreme_4hEMA34Trend_Volume_v2"
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
    
    # Get 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for Williams %R calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r_1d[highest_high_14 == lowest_low_14] = -50.0
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 and EMA89 for trend filter
    close_series_4h = pd.Series(close_4h)
    ema_34_4h = close_series_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_4h = close_series_4h.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d Williams %R to 4h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Align 4h EMA indicators to 4h timeframe (wait for completed 4h bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    ema_89_aligned = align_htf_to_ltf(prices, df_4h, ema_89_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) with EMA34 > EMA89 and volume confirmation
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with EMA34 < EMA89 and volume confirmation
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought territory)
            if williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold territory)
            if williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals