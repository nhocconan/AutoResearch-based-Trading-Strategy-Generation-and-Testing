#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Fade_1dTrend_Volume
# Hypothesis: Fade extreme Camarilla levels (R3/S3) in direction of daily trend with volume confirmation.
# In ranging markets (60% of time), price reverts from R3/S3. In trending markets, avoid fading.
# Daily EMA50 filter ensures we only take mean-reversion trades when higher timeframe is ranging.
# Volume spike confirms institutional interest at extremes. Designed for low frequency (~15-25/year).

name = "6h_Camarilla_R3_S3_Fade_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # === Daily Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    daily_high_1d = df_1d['high'].values
    daily_low_1d = df_1d['low'].values
    
    # Daily EMA50 for trend filter (ranging vs trending)
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Camarilla Levels from Previous Day ===
    # Calculate from previous day's OHLC (shifted by 1 to avoid lookahead)
    prev_close = np.roll(daily_close_1d, 1)
    prev_high = np.roll(daily_high_1d, 1)
    prev_low = np.roll(daily_low_1d, 1)
    # First value remains 0 (no previous day)
    prev_close[0] = 0
    prev_high[0] = 0
    prev_low[0] = 0
    
    # Camarilla multipliers
    R3 = prev_close + (prev_high - prev_low) * 1.1000
    S3 = prev_close - (prev_high - prev_low) * 1.1000
    
    # Align to 6h (these levels are fixed for the entire day)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    
    # === Volume Spike (24-period on 6h = 6 days) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # FADE LONG: Price at S3 support + ranging market (price near daily EMA) + volume spike
            # Ranging condition: price within 1% of daily EMA50
            near_ema = abs(close[i] - ema_50_6h[i]) / ema_50_6h[i] < 0.01
            at_s3 = close[i] <= S3_6h[i] * 1.001  # Allow small buffer
            
            if at_s3 and near_ema and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # FADE SHORT: Price at R3 resistance + ranging market + volume spike
            elif close[i] >= R3_6h[i] * 0.999 and near_ema and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches midpoint or opposite extreme
            midpoint = (R3_6h[i] + S3_6h[i]) / 2
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches midpoint or opposite extreme
            midpoint = (R3_6h[i] + S3_6h[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals