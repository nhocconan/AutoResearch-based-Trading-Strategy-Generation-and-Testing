#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel (20-period high/low) from prior 1d session for breakout entries
# 1w EMA50 confirms trend direction to avoid counter-trend trades in ranging markets
# Volume confirmation (>1.5x 50 EMA) ensures breakout has participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 30-100 total trades over 4 years = 7-25/year for 1d.
# Works in both bull and bear: EMA50 ensures we only trade with the trend, Donchian provides structure.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (using 1d HTF data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min for Donchian upper/lower bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use prior completed 1d bar only (no look-ahead)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    # Align Donchian levels to 1d timeframe (completed 1d bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_shifted)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: 50-period EMA of volume on 1d timeframe
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + price above 1w EMA50 + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > (1.5 * vol_ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + price below 1w EMA50 + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > (1.5 * vol_ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below 1w EMA50
            donchian_mid = (donchian_high + donchian_low) / 2.0
            donchian_mid_shifted = np.roll(donchian_mid, 1)
            donchian_mid_shifted[0] = np.nan
            donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_shifted)
            
            if not np.isnan(donchian_mid_aligned[i]) and (close[i] < donchian_mid_aligned[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above 1w EMA50
            donchian_mid = (donchian_high + donchian_low) / 2.0
            donchian_mid_shifted = np.roll(donchian_mid, 1)
            donchian_mid_shifted[0] = np.nan
            donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_shifted)
            
            if not np.isnan(donchian_mid_aligned[i]) and (close[i] > donchian_mid_aligned[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals