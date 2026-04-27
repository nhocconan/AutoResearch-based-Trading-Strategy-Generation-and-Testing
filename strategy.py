#!/usr/bin/env python3
"""
6h_Pivot_Breakout_Scalping_1dTrend_Filter_v1
Hypothesis: Trade 6h timeframe with daily pivot breakouts (R1/S1) filtered by daily trend and volume spikes.
Uses floor-based pivots (standard calculation) for institutional levels. Designed for 15-30 trades/year per symbol.
Works in bull via breakouts above R1 and bear via breakdowns below S1, filtered by daily trend alignment.
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
    
    # Calculate DAILY PIVOT POINTS (standard floor trader pivots)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1d EMA50 for trend filter (smooth trend)
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5 * 20-period average (on 6h data, ~5 days)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for pivots (2 days), EMA (50), volume average (20)
    start_idx = max(2, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation AND above daily EMA50 (uptrend)
            if close[i] > r1_val and vol_conf and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume confirmation AND below daily EMA50 (downtrend)
            elif close[i] < s1_val and vol_conf and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (opposite level) or closes below EMA50
            if close[i] < s1_val or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R1 (opposite level) or closes above EMA50
            if close[i] > r1_val or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Pivot_Breakout_Scalping_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0