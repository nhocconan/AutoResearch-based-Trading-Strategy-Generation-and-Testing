#!/usr/bin/env python3
# 6H_1D_WilliamsVIX_Fix_Breakout_Trend
# Hypothesis: In both bull and bear markets, extreme price movements often reverse after reaching exhaustion.
# Williams VIX Fix identifies market bottoms/tops by measuring how close the low is to the highest high
# over a lookback period. We use this on daily timeframe to detect potential reversals, then enter on
# 6H breakouts in the direction of the reversal with trend confirmation. This captures mean reversion
# after panic selling or euphoric buying, which occurs in both bull and bear markets.

name = "6H_1D_WilliamsVIX_Fix_Breakout_Trend"
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
    
    # Get 1d data for Williams VIX Fix and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:  # Need enough for 22-period calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams VIX Fix: measures how close the low is to the highest high
    # Higher values indicate potential market bottoms
    # wvf = ((highest_high - low) / (highest_high - lowest_low)) * 100
    # We invert it so high values = potential bottom
    highest_high = pd.Series(high_1d).rolling(window=22, min_periods=22).max().values
    lowest_low = pd.Series(low_1d).rolling(window=22, min_periods=22).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    wvf = ((highest_high - low_1d) / rr) * 100
    
    # Signal when WVF is high (indicating potential bottom) or low (indicating potential top)
    # We'll use extreme values: above 80 for potential bottom, below 20 for potential top
    wvf_high_signal = wvf > 80   # Potential bottom - look for longs
    wvf_low_signal = wvf < 20    # Potential top - look for shorts
    
    # 1d trend filter: EMA(50) to avoid counter-trend trades in strong trends
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1d > ema_50
    
    # Align 1d indicators to 6h
    wvf_high_aligned = align_htf_to_ltf(prices, df_1d, wvf_high_signal)
    wvf_low_aligned = align_htf_to_ltf(prices, df_1d, wvf_low_signal)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(wvf_high_aligned[i]) or np.isnan(wvf_low_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: potential bottom signal + uptrend (buy the dip in uptrend)
            if wvf_high_aligned[i] and trend_up_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: potential top signal + downtrend (sell the rally in downtrend)
            elif wvf_low_aligned[i] and not trend_up_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: potential top signal or trend turns down
            if wvf_low_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: potential bottom signal or trend turns up
            if wvf_high_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals