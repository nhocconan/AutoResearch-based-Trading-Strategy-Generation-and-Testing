#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_ATRStop_v1
Hypothesis: Trade daily Camarilla R3/S3 breakouts with weekly EMA50 trend filter, volume confirmation, and ATR-based stoploss.
Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag. Uses 1w HTF for trend alignment.
Works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) for stoploss from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R3/S3)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)  # R3 = C + (H-L)*1.1/4
    s3 = pivot - (range_hl * 1.1 / 4.0)  # S3 = C - (H-L)*1.1/4
    
    # Align Camarilla levels to 1d (same timeframe as prices)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of 1w EMA(50), 1d ATR(14), volume MA(20)
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
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
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above R3 AND volume confirm AND 1w uptrend
            long_signal = (close_val > r3_aligned[i]) and vol_conf and trend_up
            
            # Short: price breaks below S3 AND volume confirm AND 1w downtrend
            short_signal = (close_val < s3_aligned[i]) and vol_conf and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # ATR trailing stop: exit if price drops 2.5 * ATR from highest since entry
            if close_val < highest_since_entry - 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: 1w trend flips down
            elif not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.5 * ATR from lowest since entry
            if close_val > lowest_since_entry + 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: 1w trend flips up
            elif not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0