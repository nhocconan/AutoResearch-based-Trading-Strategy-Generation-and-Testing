#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_v1
Hypothesis: Trade 1d Camarilla R3/S3 breakouts with 1-week EMA34 trend filter and volume confirmation.
R3/S3 levels act as stronger support/resistance than R1/S1, reducing false breakouts.
In bull markets: price breaks above R3 with 1w uptrend → long continuation.
In bear markets: price breaks below S3 with 1w downtrend → short continuation.
Volume confirmation ensures breakouts have participation.
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Using R3/S3 for breakout entries (stronger levels than R1/S1)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # Where C = (H+L+Close)/3 of previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Align Camarilla levels to 1d
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (20d average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w EMA(34), volume MA(20), and need 1d data
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above R3 AND volume confirm AND 1w uptrend
            long_signal = (close_val > r3_aligned[i]) and vol_conf and trend_up
            
            # Short: price breaks below S3 AND volume confirm AND 1w downtrend
            short_signal = (close_val < s3_aligned[i]) and vol_conf and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below S3 (failed breakout) OR 1w trend flips down
            if (close_val < s3_aligned[i]) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R3 (failed breakdown) OR 1w trend flips up
            if (close_val > r3_aligned[i]) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_v1"
timeframe = "1d"
leverage = 1.0