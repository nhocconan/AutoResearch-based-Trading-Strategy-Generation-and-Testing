#!/usr/bin/env python3
# Weekly Donchian Breakout + Daily Volume Spike + 4H EMA Trend Filter
# Enters long when price breaks above weekly Donchian high with volume spike and 4H EMA uptrend
# Enters short when price breaks below weekly Donchian low with volume spike and 4H EMA downtrend
# Uses 1d timeframe with weekly HTF for structure, filters out low-volume breakouts
# Target: 20-50 total trades over 4 years (5-12/year) with size 0.25

name = "1d_WeeklyDonchianBreakout_VolumeSpike_4HEMA"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels
    donch_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Get daily data for volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume average
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Get 4H data for EMA trend filter (50-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4H EMA50
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume spike condition (volume > 1.5x average)
            volume_spike = volume[i] > (1.5 * vol_ma_aligned[i])
            
            # Enter long: price breaks above weekly Donchian high + volume spike + 4H EMA uptrend
            if (close[i] > donch_high_aligned[i]) and volume_spike and (close[i] > ema_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + volume spike + 4H EMA downtrend
            elif (close[i] < donch_low_aligned[i]) and volume_spike and (close[i] < ema_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low OR loses 4H EMA uptrend
            if (close[i] < donch_low_aligned[i]) or (close[i] < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high OR loses 4H EMA downtrend
            if (close[i] > donch_high_aligned[i]) or (close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals