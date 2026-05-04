#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel from prior completed 4h for structure, 1d EMA50 for trend filter
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1d EMA50 ensures we only trade with the major trend, reducing whipsaw in ranging markets.
# Works in both bull and bear by following the higher timeframe trend.

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channel (20) from prior completed 4h bar
    # We need to look back 20 completed 4h bars, so we use rolling window on 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Align Donchian levels to 4h timeframe (already aligned, just need to shift for completed bar)
    # Since we're using completed 4h bars, we shift by 1 to avoid look-ahead
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + price above 1d EMA50 + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + price below 1d EMA50 + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below 1d EMA50
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] < donchian_mid or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above 1d EMA50
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] > donchian_mid or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals