#!/usr/bin/env python3
"""
1d_Camarilla_H4_Pivot_Bounce_v1
Hypothesis: On 1d timeframe, price tends to reverse at Camarilla pivot levels (H4/L4) calculated from previous day.
In ranging markets (common in 2025+), these levels act as strong support/resistance.
Uses 1w trend filter to avoid trading against major trends and volume confirmation to avoid false breakouts.
Target: 15-30 trades per year (~60-120 over 4 years) on 1d timeframe.
"""

name = "1d_Camarilla_H4_Pivot_Bounce_v1"
timeframe = "1d"
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
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA10 for trend filter
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # === Previous Day's OHLC for Camarilla Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (H, L, C)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to current 1d bars
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # need enough data for volume MA and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema10_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price touches or goes below L4 with volume, in uptrend or ranging
            if close[i] <= camarilla_l4_aligned[i] and volume_ok and ema10_1w_aligned[i] <= close[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above H4 with volume, in downtrend or ranging
            elif close[i] >= camarilla_h4_aligned[i] and volume_ok and ema10_1w_aligned[i] >= close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to midpoint or shows weakness
            midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price returns to midpoint or shows strength
            midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals