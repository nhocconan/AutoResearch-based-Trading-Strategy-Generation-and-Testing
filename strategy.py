#!/usr/bin/env python3
"""
4h_1D_Camarilla_Pivot_Breakout_With_Volume_Filter
Hypothesis: 4h price breakout above/below daily Camarilla R4/S4 levels with volume confirmation.
Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns.
Volume filter reduces false breakouts. Target: 20-40 trades/year.
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
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for today's Camarilla levels
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # First bar uses same day's values (no previous)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Calculate Camarilla levels
    range_1d = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r4 = camarilla_pp + (range_1d * 1.1 / 2)  # Resistance level 4
    camarilla_s4 = camarilla_pp - (range_1d * 1.1 / 2)  # Support level 4
    
    # Align to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-day average
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        if position == 0:
            # Long entry: price breaks above R4 with volume
            if close[i] > camarilla_r4_aligned[i] and vol_condition:
                position = 1
                signals[i] = position_size
            # Short entry: price breaks below S4 with volume
            elif close[i] < camarilla_s4_aligned[i] and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below pivot point
            if close[i] < camarilla_pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: price crosses above pivot point
            if close[i] > camarilla_pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1D_Camarilla_Pivot_Breakout_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0