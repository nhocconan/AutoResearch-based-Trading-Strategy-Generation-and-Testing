#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 1d EMA34 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND 1d EMA34 is falling AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for long, R1 for short) or reverses EMA34 direction.
Uses 1d HTF for EMA34 trend (avoids whipsaws in ranging markets). Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:
            continue
        # Get previous day's OHLC from 1d data aligned to current 4h bar
        # We need to use the 1d data that completed before current 4h bar
        # Since we're using aligned arrays, we can get the 1d values from previous completed day
        # But we need to map 4h bar to 1d bar index
        # Simpler: use rolling window on 4h data to approximate daily OHLC (not perfect but workable)
        # Better approach: calculate from actual 1d OHLC and align
        pass
    
    # Since we need actual 1d OHLC for Camarilla, let's get the full 1d OHLC data
    # We'll calculate Camarilla levels from 1d data and then align to 4h
    if len(df_1d) < 1:
        return np.zeros(n)
        
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1_1d = np.full(len(df_1d), np.nan)
    camarilla_s1_1d = np.full(len(df_1d), np.nan)
    camarilla_r2_1d = np.full(len(df_1d), np.nan)
    camarilla_s2_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            continue  # Need previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Camarilla levels
        range_val = prev_high - prev_low
        camarilla_r1_1d[i] = prev_close + range_val * 1.1 / 12
        camarilla_s1_1d[i] = prev_close - range_val * 1.1 / 12
        camarilla_r2_1d[i] = prev_close + range_val * 1.1 / 6
        camarilla_s2_1d[i] = prev_close - range_val * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2_1d)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA34 rising AND volume spike
            if price > r1 and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND EMA34 falling AND volume spike
            elif price < s1 and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1 level OR EMA34 starts falling
                if price < s1 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1 level OR EMA34 starts rising
                if price > r1 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0