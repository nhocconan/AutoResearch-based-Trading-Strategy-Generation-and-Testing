#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12-hour timeframe, use weekly Camarilla pivot levels with 1-week trend filter and volume confirmation.
Long when price touches S3 level with weekly EMA(20) trending up and volume > 1.5x 20-period average.
Short when price touches R3 level with weekly EMA(20) trending down and volume > 1.5x 20-period average.
Exit when price reaches the opposite pivot level (S3->S2->S1 for long, R3->R2->R1 for short).
Designed for 15-30 trades/year to minimize fee drift while capturing mean-reversion bounces at institutional levels.
Works in both bull/bear markets as Camarilla levels adapt to volatility and weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: based on previous week's range
    # H = high, L = low, C = close
    # S1 = C - (H-L)*1.0/6, S2 = C - (H-L)*2.0/6, S3 = C - (H-L)*3.0/6
    # R1 = C + (H-L)*1.0/6, R2 = C + (H-L)*2.0/6, R3 = C + (H-L)*3.0/6
    rng = high_1w - low_1w
    c = close_1w
    
    s3 = c - rng * 3.0 / 6.0
    s2 = c - rng * 2.0 / 6.0
    s1 = c - rng * 1.0 / 6.0
    r1 = c + rng * 1.0 / 6.0
    r2 = c + rng * 2.0 / 6.0
    r3 = c + rng * 3.0 / 6.0
    
    # Align weekly levels to 12h timeframe (using previous week's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # Weekly trend filter: EMA(20) slope
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    weekly_trend_up = np.zeros(len(ema_20_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_20_1w_aligned), dtype=bool)
    for i in range(1, len(ema_20_1w_aligned)):
        if not np.isnan(ema_20_1w_aligned[i]) and not np.isnan(ema_20_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
            weekly_trend_down[i] = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S2 level (take profit at 2/3 of range)
            if close[i] >= s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R2 level (take profit at 2/3 of range)
            if close[i] <= r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: price touches S3 level with weekly uptrend
                if (close[i] <= s3_aligned[i] * 1.001 and  # Allow small tolerance for touch
                    weekly_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R3 level with weekly downtrend
                elif (close[i] >= r3_aligned[i] * 0.999 and  # Allow small tolerance for touch
                      weekly_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals