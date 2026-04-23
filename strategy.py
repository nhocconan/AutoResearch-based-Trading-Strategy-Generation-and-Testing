#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike filter and ATR trailing stop.
Long when price breaks above Camarilla R1 AND 1d volume > 2.0x 20-period average volume (institutional participation).
Short when price breaks below Camarilla S1 AND 1d volume > 2.0x 20-period average volume.
Exit when price retraces to Camarilla PP (pivot point) OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~20-35 trades/year on 4h timeframe.
Camarilla levels provide precise intraday support/resistance; volume filter ensures breakout conviction;
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
    
    # Calculate 1d OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for 1d (based on previous day's OHLC)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # PP = (high + low + close)/3
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * range_1d / 12.0
    camarilla_s1 = close_1d - 1.1 * range_1d / 12.0
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # 1d volume average (20-period) for spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR(14) for 4h trailing stop calculation
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
    start_idx = max(20, 20)  # vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        atr_val = atr[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        
        # Current 1d volume (use last value of 1d volume array, aligned to current 4h bar)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_val = vol_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND volume spike (2.0x avg)
            if close[i] > r1_val and vol_1d_val > 2.0 * vol_ma_1d_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S1 AND volume spike
            elif close[i] < s1_val and vol_1d_val > 2.0 * vol_ma_1d_val:
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
            
            # Primary exit: Price retraces to Camarilla PP
            if position == 1 and close[i] <= pp_val:
                exit_signal = True
            elif position == -1 and close[i] >= pp_val:
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

name = "4H_Camarilla_R1S1_Breakout_1dVolumeSpike_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0