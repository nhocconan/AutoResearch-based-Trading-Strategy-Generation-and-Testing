#!/usr/bin/env python3
"""
6H_OrderFlow_Volume_Imbalance
Hypothesis: Uses volume imbalance between consecutive 6h bars to detect institutional order flow.
Strong buying pressure (positive imbalance) in uptrend or selling pressure (negative imbalance) in downtrend signals continuation.
Works in bull markets by riding institutional buying, in bear markets by following institutional distribution.
Uses 1-day trend filter to avoid counter-trend trades and reduce whipsaw.
Targets 12-30 trades/year by requiring volume imbalance + trend alignment + minimum bar size.
"""
name = "6H_OrderFlow_Volume_Imbalance"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1D EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume imbalance: (current volume - previous volume) / previous volume
    vol_change = np.diff(volume)
    vol_prev = volume[:-1]
    vol_imbalance = np.where(vol_prev != 0, vol_change / vol_prev, 0)
    vol_imbalance = np.concatenate([[0], vol_imbalance])  # align with volume index
    
    # Calculate minimum bar size filter: true range > 0.5 * ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    min_size = atr14 * 0.5
    size_filter = tr >= min_size
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: positive volume imbalance (buying pressure) + uptrend + sufficient bar size
            if (vol_imbalance[i] > 0.15 and 
                close[i] > ema_50_1d_aligned[i] and 
                size_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: negative volume imbalance (selling pressure) + downtrend + sufficient bar size
            elif (vol_imbalance[i] < -0.15 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  size_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: volume imbalance turns negative OR price breaks trend
            if (vol_imbalance[i] < -0.05 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: volume imbalance turns positive OR price breaks trend
            if (vol_imbalance[i] > 0.05 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals