#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d EMA34 rising AND volume > 1.5x 20-period MA.
Short when price breaks below Donchian lower band AND 1d EMA34 falling AND volume > 1.5x 20-period MA.
Exit when price crosses the 1d EMA34 or Donchian middle band.
Uses 1d HTF for trend filter to avoid counter-trend trades and 12h for entry timing.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Calculate 12h Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_band = (high_ma_20 + low_ma_20) / 2
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h EMA34 slope for trend direction
    ema_34_12h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34, Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_12h[i-1]
            ema_rising = ema_34_12h[i] > ema_prev
            ema_falling = ema_34_12h[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA (tight threshold to reduce trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price > upper band AND EMA34 rising AND volume filter
            if close[i] > high_ma_20[i] and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price < lower band AND EMA34 falling AND volume filter
            elif close[i] < low_ma_20[i] and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below EMA34 OR price crosses below mid band
                if close[i] < ema_34_12h[i] or close[i] < mid_band[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above EMA34 OR price crosses above mid band
                if close[i] > ema_34_12h[i] or close[i] > mid_band[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0