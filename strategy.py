#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot point (classic)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    width = prev_week_high - prev_week_low
    R1 = 2 * pivot - prev_week_low
    S1 = 2 * pivot - prev_week_high
    R2 = pivot + width
    S2 = pivot - width
    R3 = prev_week_high + 2 * (pivot - prev_week_low)
    S3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Align weekly pivots to 6h
    R1_6h = align_htf_to_ltf(prices, df_1w, R1)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1)
    R2_6h = align_htf_to_ltf(prices, df_1w, R2)
    S2_6h = align_htf_to_ltf(prices, df_1w, S2)
    R3_6h = align_htf_to_ltf(prices, df_1w, R3)
    S3_6h = align_htf_to_ltf(prices, df_1w, S3)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 20-period EMA for daily trend
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high AND above weekly R2 with volume spike in uptrend
            if (close[i] > donchian_high[i] and 
                close[i] > R2_6h[i] and 
                vol_spike[i] and 
                close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND below weekly S2 with volume spike in downtrend
            elif (close[i] < donchian_low[i] and 
                  close[i] < S2_6h[i] and 
                  vol_spike[i] and 
                  close[i] < ema20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls below weekly S1 or trend turns down
            if close[i] < S1_6h[i] or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above weekly R1 or trend turns up
            if close[i] > R1_6h[i] or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot levels act as strong support/resistance. Combining with Donchian breakout
# and daily trend filter captures institutional breakout/breakdown moves. Volume spike ensures conviction.
# Works in bull markets (breakouts above R2 in uptrend) and bear markets (breakdowns below S2 in downtrend).
# Discrete position size (0.25) minimizes churn. Target: 15-30 trades/year (60-120 over 4 years).