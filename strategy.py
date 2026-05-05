#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h volume spike and 12h EMA50 trend filter
# Long when price breaks above 6h Camarilla R3 level AND 12h volume > 1.8x 24-period average AND close > 12h EMA50
# Short when price breaks below 6h Camarilla S3 level AND 12h volume > 1.8x 24-period average AND close < 12h EMA50
# Exit when price crosses 6h Camarilla pivot point (mean reversion)
# Uses 6h primary timeframe with 12h HTF for volume confirmation and trend filter
# Volume confirmation ensures breakouts have conviction; EMA50 filter avoids counter-trend trades
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Camarilla_R3S3_Breakout_12hVolume_12hEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for volume and trend filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h volume spike filter
    vol_12h = df_12h['volume'].values
    if len(vol_12h) >= 24:
        vol_ma_24 = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
        volume_filter_12h = vol_12h > (1.8 * vol_ma_24)
    else:
        volume_filter_12h = np.zeros(len(df_12h), dtype=bool)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h filters to 6h timeframe
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 6h data ONCE before loop for Camarilla levels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate 6h Camarilla levels
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    camarilla_r3 = close_6h + (1.1 * (high_6h - low_6h) / 2)
    camarilla_s3 = close_6h - (1.1 * (high_6h - low_6h) / 2)
    camarilla_pivot = (high_6h + low_6h + close_6h) / 3  # Standard pivot point
    
    # Align Camarilla levels to 6h timeframe (same df_6h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_filter_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND above 12h EMA50
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter_12h_aligned[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND below 12h EMA50
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter_12h_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals