#!/usr/bin/env python3
"""
1d_Weekly_SuperTrend_1DTrend_v1
Concept: 1d SuperTrend (ATR=10, mult=3) trend filter with weekly trend alignment.
- Long: 1d close > SuperTrend AND weekly close > weekly EMA200 (bullish alignment)
- Short: 1d close < SuperTrend AND weekly close < weekly EMA200 (bearish alignment)
- Exit: 1d close crosses SuperTrend in opposite direction
- Position sizing: 0.25
- Target: 10-25 trades/year (40-100 total over 4 years)
- Works in bull/bear: Weekly EMA200 defines long-term trend, SuperTrend captures intermediate trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_SuperTrend_1DTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:  # Need enough data for weekly EMA200 (~420 days) and SuperTrend
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === 1d: SuperTrend calculation (ATR=10, multiplier=3) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (3 * atr)
    lower_band = hl_avg - (3 * atr)
    
    # SuperTrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
            direction[i] = -1
    
    # === Weekly: EMA200 trend filter ===
    weekly_close = df_1w['close'].values
    weekly_ema200 = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need enough data for ATR calculation
    
    for i in range(start_idx, n):
        # Get values
        supertrend_val = supertrend[i]
        close_val = close[i]
        weekly_ema200_val = weekly_ema200_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(supertrend_val) or np.isnan(close_val) or 
            np.isnan(weekly_ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above SuperTrend AND weekly close above weekly EMA200
            if close_val > supertrend_val and close_val > weekly_ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Price below SuperTrend AND weekly close below weekly EMA200
            elif close_val < supertrend_val and close_val < weekly_ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below SuperTrend
            if close_val < supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above SuperTrend
            if close_val > supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals