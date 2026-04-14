#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily pivot levels (previous day)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    prev_range = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    r3 = prev_close + (prev_range * 1.1 / 4)
    s3 = prev_close - (prev_range * 1.1 / 4)
    
    # Align pivot levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(r3_4h[i]) or
            np.isnan(s3_4h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d_aligned[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Require volume spike for entry
        if not volume_spike[i]:
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high AND above S3
            if close[i] > donch_high[i] and close[i] > s3_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low AND below R3
            elif close[i] < donch_low[i] and close[i] < r3_4h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR below S3
            if close[i] < donch_low[i] or close[i] < s3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR above R3
            if close[i] > donch_high[i] or close[i] > r3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_R3S3_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0