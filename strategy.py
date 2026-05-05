#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Williams %R extremes with 1d EMA50 trend filter and volume confirmation
# Long when 4h Williams %R < -80 (oversold) AND 1d EMA50 rising AND volume > 1.3 * avg_volume(20) on 1h
# Short when 4h Williams %R > -20 (overbought) AND 1d EMA50 falling AND volume > 1.3 * avg_volume(20) on 1h
# Exit when price crosses 1h EMA20 (mean reversion to short-term trend)
# Uses discrete sizing 0.20 to balance return and risk
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Williams %R identifies exhaustion points in both bull and bear markets
# 1d EMA50 filter ensures we trade with higher timeframe trend, reducing whipsaw
# Volume confirmation validates exhaustion is genuine
# Session filter (08-20 UTC) reduces noise trades

name = "1h_4hWilliamsR_1dEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:  # Need at least 14 completed 4h bars for Williams %R
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r_4h = -100 * (highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)
    
    # Align 4h Williams %R to 1h timeframe (wait for completed 4h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r_4h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h EMA20 for exit signal
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h Williams %R oversold (< -80), 1d EMA50 rising, volume confirmation, in session
            if (williams_r_aligned[i] < -80 and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h Williams %R overbought (> -20), 1d EMA50 falling, volume confirmation, in session
            elif (williams_r_aligned[i] > -20 and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1h EMA20
            if close[i] < ema_20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 1h EMA20
            if close[i] > ema_20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals