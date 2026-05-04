#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) AND 1d close > 1d EMA50 (uptrend) AND volume > 1.8x 20 EMA
# Short when price breaks below 4h Donchian lower (20) AND 1d close < 1d EMA50 (downtrend) AND volume > 1.8x 20 EMA
# Uses 4h for signal direction (structure), 1d for trend filter to avoid counter-trend trades, 1h only for entry timing precision.
# Discrete sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "1h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper and lower (20-period)
    donchian_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_10 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_10)
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 1h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 4h Donchian upper AND 1d uptrend AND volume spike
            if (close[i] > upper_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below 4h Donchian lower AND 1d downtrend AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian lower OR 1d trend changes to downtrend
            if (close[i] < lower_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian upper OR 1d trend changes to uptrend
            if (close[i] > upper_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals