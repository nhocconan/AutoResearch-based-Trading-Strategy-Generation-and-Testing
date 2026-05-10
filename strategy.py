#!/usr/bin/env python3
# 6h_Liquidity_Imbalance_With_1dTrend
# Hypothesis: During strong daily trends, liquidity imbalances (equal highs/lows) on 6h chart
# provide high-probability continuation entries. We look for:
#   - Equal highs (for shorts) or equal lows (for longs) within 0.3% tolerance
#   - In the direction of the daily trend (EMA50)
#   - With volume confirmation (above 20-period MA)
# This structure captures stop hunts and liquidity sweeps that often precede continuation
# in trending markets, working in both bull (buy liquidity sweeps) and bear (sell liquidity sweeps).

name = "6h_Liquidity_Imbalance_With_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Detect equal highs/lows (liquidity levels)
    # Equal highs: current high within 0.3% of a recent high (lookback 20 bars)
    # Equal lows: current low within 0.3% of a recent low (lookback 20 bars)
    lookback = 20
    tolerance = 0.003  # 0.3%
    
    equal_high = np.zeros(n, dtype=bool)
    equal_low = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Check equal high: current high vs recent highs
        recent_highs = high[i-lookback:i]
        max_recent_high = np.max(recent_highs)
        if abs(high[i] - max_recent_high) / max_recent_high <= tolerance:
            equal_high[i] = True
        
        # Check equal low: current low vs recent lows
        recent_lows = low[i-lookback:i]
        min_recent_low = np.min(recent_lows)
        if abs(low[i] - min_recent_low) / min_recent_low <= tolerance:
            equal_low[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_1d (50), volume MA (20), lookback (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: daily uptrend + equal low (liquidity sweep) + volume
            if uptrend and equal_low[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: daily downtrend + equal high (liquidity sweep) + volume
            elif downtrend and equal_high[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or opposite liquidity sweep
            if not uptrend or equal_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or opposite liquidity sweep
            if not downtrend or equal_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals