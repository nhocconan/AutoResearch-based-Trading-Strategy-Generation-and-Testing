#!/usr/bin/env python3
# 6h_WeeklyPivot_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Trade 6h breakouts of weekly R3/S3 levels (stronger support/resistance) aligned with daily EMA50 trend and volume confirmation.
# Weekly R3/S3 act as major barriers; breakouts with trend and volume indicate strong momentum.
# Designed for low frequency (15-30 trades/year) to survive both bull and bear markets by following higher timeframe structure.

name = "6h_WeeklyPivot_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Weekly pivot levels (R3, S3) ===
    # Calculate from weekly OHLC (previous completed week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Use previous week's OHLC for current week's pivot
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Shift by 1 to use previous week's data
    wk_high_prev = np.roll(wk_high, 1)
    wk_low_prev = np.roll(wk_low, 1)
    wk_close_prev = np.roll(wk_close, 1)
    wk_high_prev[0] = np.nan
    wk_low_prev[0] = np.nan
    wk_close_prev[0] = np.nan
    
    pivot = (wk_high_prev + wk_low_prev + wk_close_prev) / 3.0
    r3 = pivot + 2.0 * (wk_high_prev - wk_low_prev)
    s3 = pivot - 2.0 * (wk_high_prev - wk_low_prev)
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # === Volume confirmation (24-period average) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r3_aligned[i]
        breakout_down = close[i] < s3_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: breakout above R3, uptrend, volume confirmation
            if breakout_up and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below S3, downtrend, volume confirmation
            elif breakout_down and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below S3 or trend reversal
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout above R3 or trend reversal
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals