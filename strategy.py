#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel (20-period high) AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-day average volume.
Short when price breaks below lower Donchian channel (20-period low) AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-day average volume.
Exit when price reverts to the 20-day midpoint (mean of upper/lower channel) or trend reverses (price crosses 1w EMA50).
Uses daily timeframe to target ~10-25 trades/year, minimizing fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring trend confirmation via 1w EMA50 for breakout entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channel calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    midpoint_channel = (upper_channel + lower_channel) / 2.0
    
    # Calculate 20-day average volume on 1d
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels and volume MA to 1d timeframe (no alignment needed as we're on 1d)
    upper_channel_aligned = upper_channel
    lower_channel_aligned = lower_channel
    midpoint_aligned = midpoint_channel
    vol_ma_aligned = vol_ma_20
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 for 1w trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(vol_ma_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_channel_aligned[i]
        lower_val = lower_channel_aligned[i]
        midpoint_val = midpoint_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_current = volume_1d[i]  # Use 1d volume for volume confirmation
        price = close_1d[i]
        
        if position == 0:
            # Long: price breaks above upper channel AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > upper_val and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < lower_val and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to midpoint OR price breaks below 1w EMA50 (trend reversal)
                if price <= midpoint_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to midpoint OR price breaks above 1w EMA50 (trend reversal)
                if price >= midpoint_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Volume_Breakout"
timeframe = "1d"
leverage = 1.0