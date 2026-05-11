#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Bounce_TrendFilter
Hypothesis: Camarilla pivot levels (R3/S3) from 1d act as strong support/resistance.
- Long when: price touches S3 and closes back above it, 1d EMA34 uptrend, volume > 20-period average
- Short when: price touches R3 and closes back below it, 1d EMA34 downtrend, volume > 20-period average
- Exit when price reaches opposite Camarilla level (S1 for long, R1 for short) or trend reverses
Works in both bull and bear: bounces off strong intraday levels with trend filter avoids counter-trend traps.
Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
"""

name = "12h_1d_Camarilla_Pivot_Bounce_TrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels from 1d OHLC ---
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    typical_price_arr = typical_price.values
    range_1d = df_1d['high'] - df_1d['low']
    range_arr = range_1d.values
    
    # Camarilla multipliers
    R3 = typical_price_arr + 1.1 * range_arr * 1.1 / 2
    R2 = typical_price_arr + 1.1 * range_arr * 1.1 / 4
    R1 = typical_price_arr + 1.1 * range_arr * 1.1 / 6
    S1 = typical_price_arr - 1.1 * range_arr * 1.1 / 6
    S2 = typical_price_arr - 1.1 * range_arr * 1.1 / 4
    S3 = typical_price_arr - 1.1 * range_arr * 1.1 / 2
    
    # Align Camarilla levels to 12h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 34  # for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema34_1d_aligned[i]
        trend_down = close_12h[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for bounces off S3/R3 with trend and volume
            if (low_12h[i] <= S3_aligned[i] and close_12h[i] > S3_aligned[i] and 
                trend_up and vol_ok):
                # Bounce off S3: long
                signals[i] = 0.25
                position = 1
            elif (high_12h[i] >= R3_aligned[i] and close_12h[i] < R3_aligned[i] and 
                  trend_down and vol_ok):
                # Rejection at R3: short
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: reach S1 (target) or trend turns down
                if close_12h[i] >= S1_aligned[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: reach R1 (target) or trend turns up
                if close_12h[i] <= R1_aligned[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals