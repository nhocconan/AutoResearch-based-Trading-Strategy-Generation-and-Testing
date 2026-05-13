#!/usr/bin/env python3
# 6h_Weekly_Pivot_Trend_Reversal_With_Volume
# Hypothesis: Mean-reversion from weekly pivot points (R4/S4) with 1d trend filter and volume spike confirmation. 
# In bull markets, buy near S3/S4 with 1d uptrend; in bear markets, sell near R3/R4 with 1d downtrend.
# Uses 6h timeframe to limit trades (target: 50-150 over 4 years) while capturing multi-day reversals.
# Volume > 2x 20-period average confirms genuine institutional interest at extremes.

name = "6h_Weekly_Pivot_Trend_Reversal_With_Volume"
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivots from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    pivot_point = np.full_like(prev_close, np.nan)
    r1 = np.full_like(prev_close, np.nan)
    r2 = np.full_like(prev_close, np.nan)
    r3 = np.full_like(prev_close, np.nan)
    r4 = np.full_like(prev_close, np.nan)
    s1 = np.full_like(prev_close, np.nan)
    s2 = np.full_like(prev_close, np.nan)
    s3 = np.full_like(prev_close, np.nan)
    s4 = np.full_like(prev_close, np.nan)
    
    pivot_point[valid_idx] = (prev_high[valid_idx] + prev_low[valid_idx] + prev_close[valid_idx]) / 3
    r1[valid_idx] = 2 * pivot_point[valid_idx] - prev_low[valid_idx]
    s1[valid_idx] = 2 * pivot_point[valid_idx] - prev_high[valid_idx]
    r2[valid_idx] = pivot_point[valid_idx] + (prev_high[valid_idx] - prev_low[valid_idx])
    s2[valid_idx] = pivot_point[valid_idx] - (prev_high[valid_idx] - prev_low[valid_idx])
    r3[valid_idx] = prev_high[valid_idx] + 2 * (pivot_point[valid_idx] - prev_low[valid_idx])
    s3[valid_idx] = prev_low[valid_idx] - 2 * (prev_high[valid_idx] - pivot_point[valid_idx])
    r4[valid_idx] = prev_high[valid_idx] + 3 * (pivot_point[valid_idx] - prev_low[valid_idx])
    s4[valid_idx] = prev_low[valid_idx] - 3 * (prev_high[valid_idx] - pivot_point[valid_idx])
    
    # Align weekly pivots to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # 6-bar cooldown to prevent overtrading
    
    for i in range(100, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Price near S4 with 1d uptrend and volume spike
            if (s4_aligned[i] > 0 and not np.isnan(s4_aligned[i]) and 
                low[i] <= s4_aligned[i] * 1.005 and  # Within 0.5% of S4
                close[i] > ema_50_1d_aligned[i] and   # 1d uptrend
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near R4 with 1d downtrend and volume spike
            elif (r4_aligned[i] > 0 and not np.isnan(r4_aligned[i]) and 
                  high[i] >= r4_aligned[i] * 0.995 and  # Within 0.5% of R4
                  close[i] < ema_50_1d_aligned[i] and   # 1d downtrend
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R2 or trend breaks
            if (r2_aligned[i] > 0 and not np.isnan(r2_aligned[i]) and 
                high[i] >= r2_aligned[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 6  # 6-bar cooldown
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S2 or trend breaks
            if (s2_aligned[i] > 0 and not np.isnan(s2_aligned[i]) and 
                low[i] <= s2_aligned[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 6  # 6-bar cooldown
            else:
                signals[i] = -0.25
    
    return signals