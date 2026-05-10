#!/usr/bin/env python3
# 6h_Liquidity_Sweep_Reversal_1wTrend
# Hypothesis: On 6h timeframe, price often sweeps liquidity (recent swing high/low) before reversing in direction of weekly trend.
# Uses 1-week EMA50 for trend filter and 6-hour swing high/low for liquidity sweep detection.
# Entry: Price breaks recent swing point (liquidity sweep) then closes back inside range, aligned with weekly trend.
# Designed for low frequency (15-35 trades/year) to minimize fee drag and work in both bull/bear markets via trend filter.

name = "6h_Liquidity_Sweep_Reversal_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h swing high/low (lookback 20 periods ~ 5 days)
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, len(high)):
        highest_high[i] = np.max(high[i - lookback:i])
        lowest_low[i] = np.min(low[i - lookback:i])
    
    # Align weekly trend to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)  # Ensure enough history
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Liquidity sweep long: sweep below recent low, then close back above it, in uptrend
            if low[i] < lowest_low[i] and close[i] > lowest_low[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Liquidity sweep short: sweep above recent high, then close back below it, in downtrend
            elif high[i] > highest_high[i] and close[i] < highest_high[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below recent low or weekly trend turns down
            if low[i] < lowest_low[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above recent high or weekly trend turns up
            if high[i] > highest_high[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals