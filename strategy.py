#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume_Filter
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) as support/resistance on 12h chart.
Enters long when price breaks above R1 with 1d EMA34 uptrend and volume spike,
enters short when price breaks below S1 with 1d EMA34 downtrend and volume spike.
Exits when price returns to the pivot point (P) or trend weakens.
Designed for ~20-30 trades/year on 12h to avoid fee drag while capturing institutional pivot levels.
Works in both bull/bear markets via trend filter and volume confirmation.
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
    
    # === DAILY DATA (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using typical price: (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Previous day's values for today's pivot (lookback 1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Pivot point (P)
    P = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    R1 = P + 1.1 * (prev_high - prev_low) / 12
    S1 = P - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 12h timeframe (wait for daily close)
    P_aligned = align_htf_to_ltf(prices, df_1d, P.values)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close']
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h INDICATORS ===
    # Volume spike: current > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 warmup
    
    for i in range(start_idx, n):
        if (np.isnan(P_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            if close[i] > R1_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with downtrend and volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to pivot point or trend weakens
            if close[i] <= P_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to pivot point or trend weakens
            if close[i] >= P_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume_Filter"
timeframe = "12h"
leverage = 1.0