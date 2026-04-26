#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Donchian(20) breakouts on 12h with 1d EMA34 trend filter and volume confirmation. 
The Donchian channel identifies strong support/resistance levels based on 20-bar highs/lows. 
Breakouts above/below these levels with volume and 1d trend alignment capture sustained moves 
while avoiding whipsaws. The 1d EMA34 provides a slower trend filter that works in both bull 
and bear markets. Targeting 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) on 12h for stoploss reference
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels on 12h data
    # We need to calculate on 12h data, then align to 12h prices
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_roll = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    dch_high_12h = align_htf_to_ltf(prices, df_12h, high_roll)
    dch_low_12h = align_htf_to_ltf(prices, df_12h, low_roll)
    
    # Volume spike detection: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(dch_high_12h[i]) or
            np.isnan(dch_low_12h[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Long logic: price breaks above Donchian high with volume spike + in uptrend
        if close[i] > dch_high_12h[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below Donchian low with volume spike + in downtrend
        elif close[i] < dch_low_12h[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Donchian level or trend weakens
        elif position == 1 and (close[i] < dch_low_12h[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > dch_high_12h[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0