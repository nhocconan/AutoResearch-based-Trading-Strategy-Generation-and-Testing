#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Fade_1dTrend_Volume
# Hypothesis: 6s mean reversion at Camarilla R3/S3 levels with 1d trend filter and volume confirmation.
# Long when: price touches or crosses below S3, 1d EMA50 rising, volume spike.
# Short when: price touches or crosses above R3, 1d EMA50 falling, volume spike.
# Exit when price reverts to Camarilla pivot (H4/L4) or 1d trend reverses.
# Works in range-bound markets (reversion at extremes) and trending markets (breakouts fade before resumption).

name = "6h_Camarilla_R3_S3_Fade_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA50 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema_1d[i] = np.mean(close_1d[0:50])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_1d[i-1] * (49 / (50 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_1d), np.nan)
    for i in range(51, len(close_1d)):
        ema_slope[i] = ema_1d[i] - ema_1d[i-1]
    
    # Align 1d EMA and slope to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # --- Camarilla levels from 1d (based on prior day) ---
    # We use the prior day's OHLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_pivot = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)  # Exit level
    camarilla_l4 = np.full(len(close_1d), np.nan)  # Exit level
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        camarilla_pivot[i] = (prev_high + prev_low + prev_close) / 3
        camarilla_r3[i] = camarilla_pivot[i] + range_ * 1.1 / 4
        camarilla_s3[i] = camarilla_pivot[i] - range_ * 1.1 / 4
        camarilla_h4[i] = camarilla_pivot[i] + range_ * 1.1 / 2
        camarilla_l4[i] = camarilla_pivot[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA50 and volume MA(20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            # Fade at S3 (long) or R3 (short) with trend filter and volume
            if close[i] <= s3_aligned[i] and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: price at S3, 1d uptrend, volume spike
                signals[i] = 0.25
                position = 1
            elif close[i] >= r3_aligned[i] and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: price at R3, 1d downtrend, volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price reaches H4 (Camarilla resistance) OR trend turns down
                if close[i] >= h4_aligned[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches L4 (Camarilla support) OR trend turns up
                if close[i] <= l4_aligned[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals