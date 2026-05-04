#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Donchian breakouts capture strong momentum moves
# 1d EMA34 ensures we only trade in the direction of the daily trend
# Volume spike confirms institutional participation
# Works in both bull and bear markets due to trend filter + volume confirmation
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Get 1d data for Donchian(20) calculation (using prior completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: upper = max(high, 20), lower = min(low, 20)
    # Using prior completed 1d bar to avoid look-ahead
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 1d bar (donchian from t-1)
    high_20_shifted = np.roll(high_20, 1)
    low_20_shifted = np.roll(low_20, 1)
    high_20_shifted[0] = np.nan
    low_20_shifted[0] = np.nan
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high AND 1d EMA34 uptrend AND volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low AND 1d EMA34 downtrend AND volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR below 1d EMA34
            if close[i] < donchian_low_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR above 1d EMA34
            if close[i] > donchian_high_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals