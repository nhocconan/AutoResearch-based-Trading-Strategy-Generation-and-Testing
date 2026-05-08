#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and 1w trend filter
# Uses 1d Donchian(20) breakout for entry, 1d volume spike for confirmation, and 1w EMA50 for trend filter
# Designed for low trade frequency (15-30/year) to avoid fee drag on 4h timeframe
# Works in both bull/bear markets by following weekly trend and breakout momentum

name = "4h_Donchian1d_Volume_1wTrend"
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
    
    # Get 1d data for Donchian breakout and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (period=20)
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1d volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ema * 2.0)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w = close_1w > ema_50  # 1 for uptrend, 0 for downtrend
    
    # Align indicators to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_spike_4h = align_htf_to_ltf(prices, df_1d, vol_spike)
    trend_1w_4h = align_htf_to_ltf(prices, df_1w, trend_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_4h[i]) or 
            np.isnan(donchian_low_4h[i]) or 
            np.isnan(vol_spike_4h[i]) or 
            np.isnan(trend_1w_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + volume spike + 1w uptrend
            if close[i] > donchian_high_4h[i] and vol_spike_4h[i] and trend_1w_4h[i] == 1:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + volume spike + 1w downtrend
            elif close[i] < donchian_low_4h[i] and vol_spike_4h[i] and trend_1w_4h[i] == 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donchian_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donchian_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals