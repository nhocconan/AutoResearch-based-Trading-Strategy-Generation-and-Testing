#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 is rising AND volume > 2.0x 30-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 is falling AND volume > 2.0x 30-period average.
Exit when price touches the opposite Camarilla level (S3 for longs, R3 for shorts) or reverses EMA34 direction.
Uses 1d HTF for EMA34 trend to avoid whipsaws in ranging markets. Target: 75-150 total trades over 4 years (19-37/year).
Camarilla R3/S3 levels are wider than R1/S1, reducing trade frequency while capturing stronger breaks.
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
    
    # Calculate 1d Camarilla levels from previous 1d OHLC
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using that day's OHLC)
    rang = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * (rang / 12) * 6  # R3 = C + 1.1*(H-L)/12*6
    camarilla_s3 = close_1d - 1.1 * (rang / 12) * 6  # S3 = C - 1.1*(H-L)/12*6
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3, additional_delay_bars=1)
    
    # 30-period volume average for spike filter (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 30)  # EMA34 (34), volume MA (30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
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
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume spike
            if price > r3 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.30
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume spike
            elif price < s3 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S3 OR EMA34 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 OR EMA34 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_LevelExit"
timeframe = "4h"
leverage = 1.0