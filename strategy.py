#!/usr/bin/env python3
"""
6h_Pivot_Fade_Trend
Hypothesis: Fade extreme daily pivot levels (R4/S4) with 1w trend filter on 6h chart.
In ranging markets, price tends to revert from extreme pivot levels (R4/S4).
In trending markets, only take fades in direction of 1w trend to avoid counter-trend trades.
Uses daily Camarilla pivot levels calculated from prior day's OHLC.
Targets 15-30 trades/year to minimize fee drag while capturing mean reversion edge.
Works in both bull/bear markets via trend filter and pivot-based mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_Fade_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R4, S4) from prior day
    # R4 = close + 1.5 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # Using prior day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    R4 = prev_close + 1.5 * (prev_high - prev_low)
    S4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align pivot levels to 6h timeframe (each 6h bar gets prior day's levels)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Weekly trend filter: 21-period EMA
    ema_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or goes below S4 (support) + weekly uptrend + volume
            if (close[i] <= S4_6h[i] and 
                ema_1w_aligned[i] > ema_1w_aligned[i-1] and 
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R4 (resistance) + weekly downtrend + volume
            elif (close[i] >= R4_6h[i] and 
                  ema_1w_aligned[i] < ema_1w_aligned[i-1] and 
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to midpoint (pivot) or weekly trend turns down
            pivot = (R4_6h[i] + S4_6h[i]) / 2  # Midpoint between R4 and S4
            if close[i] >= pivot or ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to midpoint or weekly trend turns up
            pivot = (R4_6h[i] + S4_6h[i]) / 2
            if close[i] <= pivot or ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals