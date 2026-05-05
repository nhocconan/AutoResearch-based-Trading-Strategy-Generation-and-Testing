#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme reversal with 1h EMA50 trend filter and volume confirmation
# Long when 1d Williams %R < -80 (oversold) AND 1h EMA50 > previous 1h EMA50 (uptrend) AND volume > 1.8 * avg_volume(20) on 4h
# Short when 1d Williams %R > -20 (overbought) AND 1h EMA50 < previous 1h EMA50 (downtrend) AND volume > 1.8 * avg_volume(20) on 4h
# Exit when 1d Williams %R crosses back through -50 (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R on 1d provides reliable reversal signals in ranging/weak trending markets
# 1h EMA50 filter ensures we trade with the intermediate trend, reducing whipsaw
# Volume confirmation (1.8x) validates reversal strength without being too restrictive

name = "4h_1dWilliamsR_Extreme_1hEMA50_VolumeConfirm"
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
    if len(df_1d) < 14:  # Need at least 14 completed daily bars for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Get 1h data ONCE before loop for EMA50 trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:  # Need at least 50 completed hourly bars for EMA50
        return np.zeros(n)
    close_1h = df_1h['close'].values
    
    # Calculate 1h EMA50
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_50_1h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), 1h EMA50 rising (uptrend), volume confirmation, in session
            if (williams_r_1d_aligned[i] < -80.0 and 
                ema_50_1h_aligned[i] > ema_50_1h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), 1h EMA50 falling (downtrend), volume confirmation, in session
            elif (williams_r_1d_aligned[i] > -20.0 and 
                  ema_50_1h_aligned[i] < ema_50_1h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion)
            if williams_r_1d_aligned[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion)
            if williams_r_1d_aligned[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals