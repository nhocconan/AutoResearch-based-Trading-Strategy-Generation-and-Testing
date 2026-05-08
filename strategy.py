#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mdf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Exponential Moving Average (EMA) trend filter,
# 4-hour Donchian channel breakout, and 4-hour volume spike confirmation.
# Trend direction from daily EMA(34) avoids whipsaw in sideways markets.
# Breakouts are taken only in direction of daily trend with volume confirmation
# to filter false breakouts. Designed for low trade frequency (~25-40/year) to
# minimize fee drag while capturing strong momentum moves in both bull and bear markets.

name = "4h_EMA34Trend_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # 4h volume spike: 1.5x 20-period EMA
    vol_ema = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_4h > (vol_ema * 1.5)
    
    # Align 4h indicators to lower timeframe (4h to 4h is identity, but use for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above daily EMA34 + breaks above 4h Donchian high + volume spike
            if close[i] > ema_34_1d_aligned[i] and close[i] > donchian_high_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below daily EMA34 + breaks below 4h Donchian low + volume spike
            elif close[i] < ema_34_1d_aligned[i] and close[i] < donchian_low_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals