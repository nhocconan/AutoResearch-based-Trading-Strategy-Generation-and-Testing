#!/usr/bin/env python3
# 6h_12h_vwap_reversion_v1
# Strategy: 6h VWAP mean reversion with 12h trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Price reverts to VWAP in range-bound markets. 12h VWAP trend filter ensures trades align with higher timeframe momentum, reducing counter-trend trades. Works in bull/bear by fading deviations from VWAP only when higher timeframe trend is weak or ranging.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_vwap_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP for 6h
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # Deviation from VWAP as percentage
    dev_pct = (close - vwap) / vwap
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h VWAP and its slope for trend filter
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_num_12h = np.cumsum(typical_price_12h * df_12h['volume'])
    vwap_den_12h = np.cumsum(df_12h['volume'])
    vwap_12h = vwap_num_12h / vwap_den_12h
    
    # 12h VWAP slope (20-period linear regression slope proxy: difference)
    vwap_slope = vwap_12h - np.roll(vwap_12h, 20)
    vwap_slope[:20] = np.nan
    
    # Align 12h VWAP and slope to 6h
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h.values)
    vwap_slope_aligned = align_htf_to_ltf(prices, df_12h, vwap_slope.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(dev_pct[i]) or np.isnan(vwap_12h_aligned[i]) or np.isnan(vwap_slope_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: weak trend when |slope| is small
        weak_trend = abs(vwap_slope_aligned[i]) < (vwap_12h_aligned[i] * 0.001)  # 0.1% of VWAP
        
        # Mean reversion thresholds
        long_threshold = -0.008  # -0.8% deviation
        short_threshold = 0.008   # +0.8% deviation
        
        # Entry conditions: fade VWAP deviation only in weak trend
        if weak_trend and dev_pct[i] < long_threshold and position != 1:
            position = 1
            signals[i] = 0.25
        elif weak_trend and dev_pct[i] > short_threshold and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to VWAP or trend strengthens
        elif position == 1 and (dev_pct[i] > -0.002 or not weak_trend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (dev_pct[i] < 0.002 or not weak_trend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals