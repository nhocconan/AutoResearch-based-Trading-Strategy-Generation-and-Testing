#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses Donchian channel (20-period high/low) from 1d for institutional breakout structure
# 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.5x average volume) - balanced to target trade frequency
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "1d_Donchian20_1wEMA50_VolumeConfirm_Balanced_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for Donchian(20) calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Calculate Donchian channels (20-period) from prior completed 1d bar
    # Upper channel = 20-period high, Lower channel = 20-period low
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 1d bar
    upper_shifted = np.roll(upper_channel, 1)
    lower_shifted = np.roll(lower_channel, 1)
    upper_shifted[0] = np.nan
    lower_shifted[0] = np.nan
    
    # Align to 1d timeframe (no additional shift needed for 1d->1d)
    upper_aligned = upper_shifted
    lower_aligned = lower_shifted
    
    # Volume confirmation: 20-period EMA of volume from prior completed 1d bar
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_shifted = np.roll(vol_ema_20, 1)
    vol_ema_20_shifted[0] = np.nan
    vol_ema_20_aligned = vol_ema_20_shifted  # 1d->1d alignment
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND price > 1w EMA50 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel AND price < 1w EMA50 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower channel OR price crosses below 1w EMA50
            if close[i] < lower_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper channel OR price crosses above 1w EMA50
            if close[i] > upper_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals