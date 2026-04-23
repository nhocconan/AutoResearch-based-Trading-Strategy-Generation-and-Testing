#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w volume spike filter and ATR trailing stop.
Long when price breaks above 1d Donchian upper AND 1w volume > 2.0x 20-period average volume.
Short when price breaks below 1d Donchian lower AND 1w volume > 2.0x 20-period average volume.
Exit when price retraces to 1d Donchian midpoint OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting 30-100 total trades over 4 years (7-25/year).
Donchian channels provide robust structure; volume filter ensures breakout conviction;
works in expansion markets (both bull breakouts and bear breakdowns).
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
    
    # Calculate 1d OHLC for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian levels for 1d (20-period)
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    midpoint_1d = (upper_1d + lower_1d) / 2.0
    
    # Align Donchian levels to 1d (completed 1d bar only)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    # 1w volume average (20-period) for spike filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # ATR(14) for 1d trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian and vol MA need 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(midpoint_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_1w_val = vol_ma_1w_aligned[i]
        atr_val = atr[i]
        upper_val = upper_1d_aligned[i]
        lower_val = lower_1d_aligned[i]
        midpoint_val = midpoint_1d_aligned[i]
        
        # Current 1w volume (use last value of 1w volume array, aligned to current 1d bar)
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
        vol_1w_val = vol_1w_aligned[i]
        
        if position == 0:
            # Long: Break above 1d Donchian upper AND volume spike (2.0x avg)
            if close[i] > upper_val and vol_1w_val > 2.0 * vol_ma_1w_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below 1d Donchian lower AND volume spike
            elif close[i] < lower_val and vol_1w_val > 2.0 * vol_ma_1w_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 1d Donchian midpoint
            if position == 1 and close[i] <= midpoint_val:
                exit_signal = True
            elif position == -1 and close[i] >= midpoint_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wVolumeSpike_VolumeConfirmation_MidpointExit_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0