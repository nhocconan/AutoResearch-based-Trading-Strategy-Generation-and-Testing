#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price > upper Donchian(20) AND 1d EMA34 rising AND volume > 1.5x 20-period MA.
Short when price < lower Donchian(20) AND 1d EMA34 falling AND volume > 1.5x 20-period MA.
Exit when price crosses the 1d EMA34 or opposite Donchian band.
Uses 1d HTF for trend filter to avoid counter-trend trades in bear markets.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) channels
    lookback = 20
    upper_donch = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_donch = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 20)  # Donchian, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1d_aligned[i-1]
            ema_rising = ema_34_1d_aligned[i] > ema_prev
            ema_falling = ema_34_1d_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA (tight threshold to reduce trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price > upper Donchian AND EMA34 rising AND volume filter
            if close[i] > upper_donch[i] and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian AND EMA34 falling AND volume filter
            elif close[i] < lower_donch[i] and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below EMA34 OR price < lower Donchian
                if close[i] < ema_34_1d_aligned[i] or close[i] < lower_donch[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above EMA34 OR price > upper Donchian
                if close[i] > ema_34_1d_aligned[i] or close[i] > upper_donch[i]:
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