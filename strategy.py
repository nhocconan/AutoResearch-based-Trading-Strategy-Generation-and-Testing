#!/usr/bin/env python3
"""
6h_LongOnly_Trend_Retracement
Hypothesis: In strong 12h uptrends (price > EMA50), 6h retracements to EMA21 offer high-probability long entries with favorable risk-reward. Works in bull markets via buying dips in uptrends and avoids shorting in bear markets (flat only). Uses 12h EMA50 for trend filter and 6h EMA21 for entry timing, with volume confirmation to avoid false signals. Designed for low trade frequency (target: 50-150 trades over 4 years) to minimize fee drift. Long-only to avoid whipsaw in bear markets.
"""

name = "6h_LongOnly_Trend_Retracement"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA21 for entry
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6h EMA50 for trend confirmation (optional)
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x 12h average volume (scaled to 6h)
    # 12h = 2 x 6h bars, so scale 12h volume to 6h equivalent
    vol_12h_avg = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_12h_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_avg)
    vol_6h_equiv = vol_12h_avg_aligned / 2.0
    volume_filter = volume > vol_12h_avg_aligned * 1.5  # Compare raw 12h avg volume to current 6h volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Warmup: need EMA21 (21) and 12h EMA50 (50)
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema21[i]) or
            np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend filter: only trade in uptrend
        uptrend_12h = close[i] > ema50_12h_aligned[i]
        
        # Additional trend confirmation: price above 6h EMA50
        uptrend_6h = close[i] > ema50[i]
        
        if position == 0:
            # Long entry: price retracing to EMA21 in uptrend + volume
            # Condition: close <= EMA21 * 1.01 (within 1% above EMA21) and uptrend
            if close[i] <= ema21[i] * 1.01 and uptrend_12h and uptrend_6h and volume_filter[i]:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Long exit: price breaks above EMA21 significantly or trend fails
            if close[i] > ema21[i] * 1.05 or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals