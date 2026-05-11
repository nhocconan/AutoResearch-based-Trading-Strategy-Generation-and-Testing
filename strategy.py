#!/usr/bin/env python3
name = "6h_Liquidity_Sweep_Reversal"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1d = ema50_1d > ema200_1d
    trend_down_1d = ema50_1d < ema200_1d
    
    # Align trend to 6h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # Volume filter: volume spike > 2x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Liquidity sweep detection: price breaks recent swing high/low then reverses
    lookback = 10  # bars to look back for swing points
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Recent swing high (max high in lookback window)
        swing_high[i] = np.max(high[i-lookback:i])
        # Recent swing low (min low in lookback window)
        swing_low[i] = np.min(low[i-lookback:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(swing_high[i]) or np.isnan(swing_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price sweeps below recent swing low (liquidity grab) then reverses up
            # with volume spike in daily uptrend
            if (low[i] < swing_low[i] and  # broke swing low (liquidity sweep)
                close[i] > swing_low[i] and  # closed back above it (reversal)
                volume[i] > 2.0 * vol_ma20[i] and  # volume spike
                trend_up_aligned[i]):  # daily uptrend filter
                signals[i] = 0.25
                position = 1
            # Short setup: price sweeps above recent swing high then reverses down
            elif (high[i] > swing_high[i] and  # broke swing high (liquidity sweep)
                  close[i] < swing_high[i] and  # closed back below it (reversal)
                  volume[i] > 2.0 * vol_ma20[i] and  # volume spike
                  trend_down_aligned[i]):  # daily downtrend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below swing low or trend changes
            if (low[i] < swing_low[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above swing high or trend changes
            if (high[i] > swing_high[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals