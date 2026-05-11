#!/usr/bin/env python3
"""
1d_Weekly_VWAP_Deviation_v1
Hypothesis: Price tends to revert to the weekly VWAP after significant deviations.
In bull markets, deviations below weekly VWAP act as support; in bear markets,
deviations above act as resistance. Uses 1d timeframe with 1w VWAP for context.
Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.
"""

name = "1d_Weekly_VWAP_Deviation_v1"
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
    
    # === 1W Data for VWAP Calculation ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for each weekly bar
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w_array = vwap_1w.values
    
    # Align weekly VWAP to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w_array)
    
    # === Daily Indicators ===
    # Daily VWAP for entry timing
    typical_price = (high + low + close) / 3
    vwap_daily = (typical_price * volume).cumsum() / volume.cumsum()
    vwap_daily_array = vwap_daily.values
    
    # Daily ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i] if i > 0 else tr[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # Need enough data for VWAP calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(vwap_daily_array[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate deviation from weekly VWAP
        deviation = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i]
        
        if position == 0:
            # Long: price significantly below weekly VWAP and above daily VWAP (bounce signal)
            if deviation < -0.02 and close[i] > vwap_daily_array[i]:
                signals[i] = 0.25
                position = 1
            # Short: price significantly above weekly VWAP and below daily VWAP (rejection signal)
            elif deviation > 0.02 and close[i] < vwap_daily_array[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to or exceeds weekly VWAP
            if close[i] >= vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price returns to or falls below weekly VWAP
            if close[i] <= vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals