#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and 1-week trend filter.
Long when price breaks above 20-period Donchian high, 1-day volume > 50-period average, and 1-week EMA50 is rising.
Short when price breaks below 20-period Donchian low, 1-day volume > 50-period average, and 1-week EMA50 is falling.
Exit when price returns to the Donchian midline or volume drops below average.
Works in both bull and bear markets by combining breakout momentum with volume confirmation and higher-timeframe trend alignment.
"""

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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Load 1-day data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # EMA slope (rising/falling)
    ema_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema_slope[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, volume confirmation, weekly EMA rising
            if close[i] > high_roll[i] and volume_1d[i] > avg_vol_1d_aligned[i] and ema_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, volume confirmation, weekly EMA falling
            elif close[i] < low_roll[i] and volume_1d[i] > avg_vol_1d_aligned[i] and ema_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midline or volume drops below average
                if close[i] < donchian_mid[i] or volume_1d[i] < avg_vol_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midline or volume drops below average
                if close[i] > donchian_mid[i] or volume_1d[i] < avg_vol_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dVolume_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0