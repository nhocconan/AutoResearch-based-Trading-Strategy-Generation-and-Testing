#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA20 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 12h EMA20 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 AND 12h EMA20 is falling AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for long, R3 for short) or reverses EMA20 direction.
Uses 12h HTF for EMA20 trend to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 12h EMA20 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We'll approximate using 24-period lookback (4h * 6 = 24h ~ 1 day)
    lookback = 24
    camarilla_h = np.full(n, np.nan)
    camarilla_l = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        # Use high/low/close of the lookback window as approximate daily OHLC
        window_high = np.max(high[i - lookback + 1:i + 1])
        window_low = np.min(low[i - lookback + 1:i + 1])
        window_close = close[i]  # current close as proxy for daily close
        
        pivot = (window_high + window_low + window_close) / 3.0
        range_val = window_high - window_low
        
        camarilla_h[i] = pivot + range_val * 1.1 / 2.0
        camarilla_l[i] = pivot - range_val * 1.1 / 2.0
        camarilla_r3[i] = pivot + range_val * 1.1 / 4.0
        camarilla_s3[i] = pivot - range_val * 1.1 / 4.0
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 20, 20)  # Camarilla (24), EMA20 (20), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_20_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA20 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_20_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA20 rising AND volume spike
            if price > r3 and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA20 falling AND volume spike
            elif price < s3 and ema_falling and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S3 band OR EMA20 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_20_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R3 band OR EMA20 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_20_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA20_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0