#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volatility (ATR-based) for Camarilla width
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.abs(high_1d[1:] - close_1d[:-1]), 
                       np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_5_1d = pd.Series(tr_1d).rolling(window=5, min_periods=5).mean().values
    
    # Previous day's close (Camarilla base)
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla levels (using ATR for dynamic width)
    # Resistance levels
    r1 = prev_close_1d + 1.1 * atr_5_1d * 1.1
    r2 = prev_close_1d + 1.1 * atr_5_1d * 1.5
    r3 = prev_close_1d + 1.1 * atr_5_1d * 2.0
    r4 = prev_close_1d + 1.1 * atr_5_1d * 2.6
    
    # Support levels
    s1 = prev_close_1d - 1.1 * atr_5_1d * 1.1
    s2 = prev_close_1d - 1.1 * atr_5_1d * 1.5
    s3 = prev_close_1d - 1.1 * atr_5_1d * 2.0
    s4 = prev_close_1d - 1.1 * atr_5_1d * 2.6
    
    # Align Camarilla levels to 12h
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current 12h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma_20
    
    # Trend filter: 50-period EMA on 12h close
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long conditions: price touches S3/S4 with volume and above EMA50
        touch_s3 = low[i] <= s3_12h[i]
        touch_s4 = low[i] <= s4_12h[i]
        volume_confirm = volume_ok[i]
        above_ema = close[i] > ema_50[i]
        
        long_signal = (touch_s3 or touch_s4) and volume_confirm and above_ema
        
        # Short conditions: price touches R3/R4 with volume and below EMA50
        touch_r3 = high[i] >= r3_12h[i]
        touch_r4 = high[i] >= r4_12h[i]
        below_ema = close[i] < ema_50[i]
        
        short_signal = (touch_r3 or touch_r4) and volume_confirm and below_ema
        
        # Exit conditions: price returns to midpoint between S1/R1 or opposite touch
        midpoint = (s1_12h[i] + r1_12h[i]) / 2
        exit_long = high[i] >= midpoint and position == 1
        exit_short = low[i] <= midpoint and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals