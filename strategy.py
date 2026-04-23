#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 4h EMA34 is rising AND volume > 1.3x 20-period average.
Short when price breaks below Camarilla S3 AND 4h EMA34 is falling AND volume > 1.3x 20-period average.
Exit when price touches the opposite Camarilla level (R3 for shorts, S3 for longs) or reverses EMA34 direction.
Uses 4h HTF for EMA34 trend (avoids whipsaws in ranging markets). Target: 60-150 total trades over 4 years (15-37/year).
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
    
    # Calculate 4h EMA34 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate daily Camarilla pivots (R3, S3) using previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 1h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), volume MA (20)
    
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
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume confirmation
            if price > r3 and ema_rising and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume confirmation
            elif price < s3 and ema_falling and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.20
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0