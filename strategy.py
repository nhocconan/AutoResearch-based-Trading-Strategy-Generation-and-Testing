#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA50 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND 4h EMA50 is falling AND volume > 1.5x 20-period average.
Exit when price touches opposite Camarilla level (S1 for longs, R1 for shorts).
Uses 4h HTF for EMA50 trend (avoids whipsaws in ranging markets). Target: 60-150 total trades over 4 years (15-37/year).
Camarilla R1/S1 levels provide tighter structure than R3/S3 for more precise entries.
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
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d Camarilla levels from previous 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using that day's OHLC)
    rang = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * (rang / 12) * 2  # R1 = C + 1.1*(H-L)/12*2
    camarilla_s1 = close_1d - 1.1 * (rang / 12) * 2  # S1 = C - 1.1*(H-L)/12*2
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels for current day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 (50), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA50 rising AND volume spike
            if price > r1 and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND EMA50 falling AND volume spike
            elif price < s1 and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1
                if price < s1:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1
                if price > r1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeConfirmation_LevelExit"
timeframe = "1h"
leverage = 1.0