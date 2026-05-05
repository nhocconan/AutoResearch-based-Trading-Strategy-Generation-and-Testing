#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extreme readings with 6h EMA34 trend filter and volume spike confirmation
# Long when 1d Williams %R < -80 (oversold) AND price > 6h EMA34 AND volume > 1.8 * avg_volume(20) on 6h
# Short when 1d Williams %R > -20 (overbought) AND price < 6h EMA34 AND volume > 1.8 * avg_volume(20) on 6h
# Exit when Williams %R returns to neutral range (-50 to -30 for longs, -70 to -50 for shorts) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe
# Williams %R identifies exhaustion points in both bull and bear markets
# 6h EMA34 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms reversal strength and reduces false signals
# Works in bull markets (oversold bounces in uptrend) and bear markets (overbought rejections in downtrend)

name = "6h_WilliamsR_Extreme_6hEMA34_VolumeSpike"
timeframe = "6h"
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
    if len(df_1d) < 14:  # Need enough for Williams %R calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL) where HH=14-period high, LL=14-period low
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when HH == LL
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 6h data ONCE before loop for EMA34 trend filter and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 6h EMA34
    close_6h_series = pd.Series(close_6h)
    ema34_6h = close_6h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_6h > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_6h[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), price above EMA34, volume confirmation, in session
            if williams_r_aligned[i] < -80 and close[i] > ema34_6h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below EMA34, volume confirmation, in session
            elif williams_r_aligned[i] > -20 and close[i] < ema34_6h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral range (-50 to -30) OR volume drops below average
            if williams_r_aligned[i] > -50 and williams_r_aligned[i] < -30 or volume_6h[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral range (-70 to -50) OR volume drops below average
            if williams_r_aligned[i] < -50 and williams_r_aligned[i] > -70 or volume_6h[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals