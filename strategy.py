#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 12h EMA50 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND 12h EMA50 is falling AND volume > 2.0x 20-period average.
Exit when price touches the opposite Camarilla level (R3 for shorts, S3 for longs) or reverses EMA50 direction.
Uses 12h HTF for EMA50 trend to reduce whipsaws. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We'll use rolling window of 24 periods (4h * 6 = 24h ≈ 1 day) to approximate daily OHLC
    lookback = 24
    camarilla_h = np.full(n, np.nan)
    camarilla_l = np.full(n, np.nan)
    camarilla_c = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window_high = np.max(high[i - lookback + 1:i + 1])
        window_low = np.min(low[i - lookback + 1:i + 1])
        window_close = close[i]
        
        camarilla_h[i] = window_high
        camarilla_l[i] = window_low
        camarilla_c[i] = window_close
        
        if i >= lookback:  # Need previous day's data for Camarilla calculation
            prev_h = camarilla_h[i-1]
            prev_l = camarilla_l[i-1]
            prev_c = camarilla_c[i-1]
            
            if not (np.isnan(prev_h) or np.isnan(prev_l) or np.isnan(prev_c)):
                range_val = prev_h - prev_l
                camarilla_h[i] = prev_c + range_val * 1.1 / 2  # R3
                camarilla_l[i] = prev_c - range_val * 1.1 / 2  # S3
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)  # Camarilla (24), EMA50 (50), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h[i]) or np.isnan(camarilla_l[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        r3 = camarilla_h[i]  # R3 level
        s3 = camarilla_l[i]  # S3 level
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
            # Long: Break above Camarilla R3 AND EMA50 rising AND volume spike
            if price > r3 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA50 falling AND volume spike
            elif price < s3 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S3 level OR EMA50 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 level OR EMA50 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0