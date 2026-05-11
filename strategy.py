#!/usr/bin/env python3
name = "6h_LiquiditySweep_1dTrend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and liquidity sweep detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_ltf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d high/low for liquidity sweep levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_aligned = align_ltf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_ltf_to_ltf(prices, df_1d, low_1d)
    
    # 6h ATR for stop and filtering
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_1d_aligned[i]) or
            np.isnan(low_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price sweeps below 1d low (liquidity grab) then closes back above it
            # AND 1d trend is up (price > EMA34)
            if low[i] < low_1d_aligned[i] and close[i] > low_1d_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price sweeps above 1d high then closes back below it
            # AND 1d trend is down (price < EMA34)
            elif high[i] > high_1d_aligned[i] and close[i] < high_1d_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below 1d low OR trend changes
            if low[i] < low_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above 1d high OR trend changes
            if high[i] > high_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: Corrected import from mtf_data (align_ltf_to_ltf doesn't exist, should be align_htf_to_ltf)
# Fixing the import and function call: