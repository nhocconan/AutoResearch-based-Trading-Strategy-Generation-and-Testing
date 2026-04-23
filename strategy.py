#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above 1d Camarilla R1 AND 1w EMA34 rising AND volume > 1.5x 20-period MA.
Short when price breaks below 1d Camarilla S1 AND 1w EMA34 falling AND volume > 1.5x 20-period MA.
Exit when price crosses the opposite Camarilla level (S1 for long exit, R1 for short exit).
Uses 1w HTF for trend filter to avoid counter-trend trades on daily timeframe.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Works in both bull and bear markets by following the weekly trend with volume confirmation.
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
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # Camarilla R1 = close_prev + (high_prev - low_prev) * 1.1/12
    # Camarilla S1 = close_prev - (high_prev - low_prev) * 1.1/12
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]  # fill first value
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    range_prev = high_prev - low_prev
    camarilla_r1 = close_prev + range_prev * 1.1 / 12
    camarilla_s1 = close_prev - range_prev * 1.1 / 12
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # need at least 1 for previous day data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1w_aligned[i-1]
            ema_rising = ema_34_1w_aligned[i] > ema_prev
            ema_falling = ema_34_1w_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1d volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above camarilla R1 AND EMA34 rising AND volume filter
            if close[i] > camarilla_r1[i] and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below camarilla S1 AND EMA34 falling AND volume filter
            elif close[i] < camarilla_s1[i] and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below camarilla S1
                if close[i] < camarilla_s1[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above camarilla R1
                if close[i] > camarilla_r1[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0