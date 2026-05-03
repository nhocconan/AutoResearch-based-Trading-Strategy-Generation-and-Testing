#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3 with volume > 1.5x 24-bar average and close > 1d EMA34 (uptrend)
# Short when price breaks below S3 with volume > 1.5x 24-bar average and close < 1d EMA34 (downtrend)
# Exit when price reverts to opposite Camarilla level (S3 for long, R3 for short) or trend fails
# Camarilla pivots provide statistically significant intraday support/resistance levels
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 1d data for Camarilla pivot calculation (use previous day's OHLC)
    df_1d_pivot = get_htf_data(prices, '1d')
    high_1d = df_1d_pivot['high'].values
    low_1d = df_1d_pivot['low'].values
    close_1d_pivot = df_1d_pivot['close'].values
    
    # Calculate Camarilla levels for 1d timeframe
    # Pivot point = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d_pivot) / 3
    r3_1d = close_1d_pivot + (high_1d - low_1d) * 1.1 / 2
    s3_1d = close_1d_pivot - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for previous day's close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d_pivot, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d_pivot, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d_pivot, s3_1d)
    
    # Volume confirmation (1.5x 24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 24) + 1  # EMA34(1d) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike and close > 1d EMA34 (uptrend)
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike and close < 1d EMA34 (downtrend)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price reverts to S3 or close < 1d EMA34 (trend failure)
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price reverts to R3 or close > 1d EMA34 (trend failure)
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals