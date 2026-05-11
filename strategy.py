#!/usr/bin/env python3
"""
4h_12h_Camarilla_R3_S3_Breakout_TrendFilter
Hypothesis: Camarilla R3/S3 breakout from 12h timeframe acts as strong support/resistance.
Combine with 12h EMA50 trend filter and volume spike for confirmation.
Only trade in direction of 12h trend to avoid counter-trend whipsaws.
Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.
Works in bull by buying breakouts above R3 in uptrend; works in bear by selling breakdowns below S3 in downtrend.
"""

name = "4h_12h_Camarilla_R3_S3_Breakout_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h Camarilla levels (R3, S3) ---
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and ranges
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    r3_12h = close_12h + range_12h * 1.1 / 4
    s3_12h = close_12h - range_12h * 1.1 / 4
    
    # --- 12h EMA50 trend ---
    ema_50_12h = np.full(len(close_12h), np.nan)
    for i in range(len(close_12h)):
        if i < 50:
            ema_50_12h[i] = np.nan
        elif i == 50:
            ema_50_12h[i] = np.mean(close_12h[0:50])
        else:
            ema_50_12h[i] = (close_12h[i] * 2 / (50 + 1)) + (ema_50_12h[i-1] * (49 / (50 + 1)))
    
    # EMA slope for trend direction
    ema_slope_50_12h = np.full(len(close_12h), np.nan)
    for i in range(51, len(close_12h)):
        ema_slope_50_12h[i] = ema_50_12h[i] - ema_50_12h[i-1]
    
    # --- 4h ATR(14) for stoploss ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (tr[i] * 1 / 14) + (atr[i-1] * 13 / 14)
    
    # --- 4h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 12h indicators to 4h
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_slope_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(12h data needs 50 bars for EMA, 14 for ATR, 20 for vol MA)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(ema_slope_50_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: break above R3 in uptrend
            if close[i] > r3_12h_aligned[i] and ema_slope_50_12h_aligned[i] > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in downtrend
            elif close[i] < s3_12h_aligned[i] and ema_slope_50_12h_aligned[i] < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price closes below EMA50 OR breaks below S3 (reversal)
                if close[i] < ema_50_12h_aligned[i] or close[i] < s3_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above EMA50 OR breaks above R3 (reversal)
                if close[i] > ema_50_12h_aligned[i] or close[i] > r3_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals