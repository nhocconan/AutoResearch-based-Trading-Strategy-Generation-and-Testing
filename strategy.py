#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Donchian channel from prior completed 12h for structure, 1d EMA34 for higher timeframe trend filter
# Volume confirmation (>2.5x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# 1d EMA34 provides stronger trend filter than 12h EMA50, reducing whipsaw in both bull and bear markets.

name = "12h_Donchian20_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Donchian channel structure
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper/lower channels on completed 12h bars
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 12h bar (avoid look-ahead)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + price above 1d EMA34 + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (2.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + price below 1d EMA34 + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (2.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below 1d EMA34
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] < donchian_mid or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above 1d EMA34
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] > donchian_mid or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals