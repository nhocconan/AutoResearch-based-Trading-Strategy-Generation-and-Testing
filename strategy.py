#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume spike (2x 20-period MA).
Long when price breaks above upper Donchian AND 1d EMA34 rising AND volume > 2x MA.
Short when price breaks below lower Donchian AND 1d EMA34 falling AND volume > 2x MA.
Exit when price touches the opposite Donchian band or EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d EMA34 slope for trend direction (rising/falling)
    ema_34_1d_prev = np.roll(ema_34_1d_aligned, 1)
    ema_34_1d_prev[0] = np.nan
    ema_rising = ema_34_1d_aligned > ema_34_1d_prev
    ema_falling = ema_34_1d_aligned < ema_34_1d_prev
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma_20)  # Volume spike: 2x MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34)  # Donchian, volume MA, EMA warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND EMA34 rising AND volume spike
            if close[i] > highest_high[i] and ema_rising[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND EMA34 falling AND volume spike
            elif close[i] < lowest_low[i] and ema_falling[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian OR EMA34 starts falling
                if close[i] < lowest_low[i] or (i > 0 and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian OR EMA34 starts rising
                if close[i] > highest_high[i] or (i > 0 and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0