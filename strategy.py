#!/usr/bin/env python3
"""
6h_SuperTrend_1wTrend_Filter
Hypothesis: Use Supertrend (ATR=10, multiplier=3) on 6h for entry signals, filtered by weekly trend (price above/below weekly EMA50). 
This combines a responsive trend-following tool (Supertrend) with a higher-timeframe trend filter to avoid counter-trend trades. 
Works in bull markets by capturing uptrends and in bear markets by avoiding longs in downtrends (and vice versa for shorts). 
Weekly EMA50 provides a smooth, reliable trend filter that reduces whipsaws. 
Target: 20-50 trades per year (80-200 over 4 years) to stay within trade frequency limits.
"""

name = "6h_SuperTrend_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Supertrend on 6h (ATR=10, multiplier=3) ===
    atr_period = 10
    multiplier = 3
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[atr_period] = np.mean(tr[1:atr_period+1])  # Seed with simple average
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend components
    supertrend = np.zeros_like(close)
    uptrend = np.ones_like(close, dtype=bool)  # Start with uptrend assumption
    
    # Set first valid value
    supertrend[atr_period] = hl2[atr_period]
    uptrend[atr_period] = True
    
    for i in range(atr_period+1, n):
        # Calculate current upper and lower bands
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Adjust bands based on previous close
        if close[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        if close[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        # Determine trend
        if close[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i]:
                lower_band[i] = lower_band[i-1]
            else:
                upper_band[i] = upper_band[i-1]
        
        # Set Supertrend value
        supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    # === Weekly Trend Filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Entry Logic ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after Supertrend warmup
    start_idx = atr_period + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend[i]) or np.isnan(ema50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Supertrend (uptrend) AND price above weekly EMA50 (bullish higher TF)
            if close[i] > supertrend[i] and close[i] > ema50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend (downtrend) AND price below weekly EMA50 (bearish higher TF)
            elif close[i] < supertrend[i] and close[i] < ema50_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Supertrend (trend reversal)
            if close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Exit short: price closes above Supertrend (trend reversal)
            if close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals