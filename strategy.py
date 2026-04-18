#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_WeeklyTrend_Volume_Filtered
Hypothesis: Combines daily Camarilla pivot breakouts with weekly trend filter and volume confirmation. 
- Entry: Price breaks above R1 with weekly bullish trend (close > weekly EMA34) and volume spike, or breaks below S1 with weekly bearish trend (close < weekly EMA34) and volume spike.
- Exit: Price returns to pivot point (PP) or weekly trend reverses.
- Weekly trend filter avoids counter-trend trades in choppy markets.
- Designed for 12-30 trades/year on 6h timeframe with selective, high-probability entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    # Pivot Point (PP) = (H + L + C) / 3
    # R1 = C + 1.1*(H - L)
    # S1 = C - 1.1*(H - L)
    # R4 = C + 2.6*(H - L)  (breakout level)
    # S4 = C - 2.6*(H - L)  (breakdown level)
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    
    PP = (H_1d + L_1d + C_1d) / 3.0
    R1 = C_1d + 1.1 * (H_1d - L_1d)
    S1 = C_1d - 1.1 * (H_1d - L_1d)
    R4 = C_1d + 2.6 * (H_1d - L_1d)
    S4 = C_1d - 2.6 * (H_1d - L_1d)
    
    # Align daily levels to 6h timeframe (wait for daily close)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Weekly trend filter: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        k = 2 / (34 + 1)
        for i in range(len(close_1w)):
            if i == 0:
                ema34_1w[i] = close_1w[i]
            else:
                ema34_1w[i] = close_1w[i] * k + ema34_1w[i-1] * (1 - k)
    
    # Align weekly EMA to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with weekly bullish trend and volume spike
            if (close[i] > R1_aligned[i] and close[i] > ema34_1w_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with weekly bearish trend and volume spike
            elif (close[i] < S1_aligned[i] and close[i] < ema34_1w_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point or weekly trend turns bearish
            if close[i] <= PP_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point or weekly trend turns bullish
            if close[i] >= PP_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_WeeklyTrend_Volume_Filtered"
timeframe = "6h"
leverage = 1.0