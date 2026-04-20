#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_Volume_Trend_v2
Concept: 4h Camarilla R1/S1 breakout with volume confirmation and EMA trend filter.
- Long: Price breaks above R1 (resistance) with volume > 1.5x average and price > EMA50
- Short: Price breaks below S1 (support) with volume > 1.5x average and price < EMA50
- Exit: Price crosses EMA50 (trend reversal) or opposite S1/R1 level touched
- Uses daily pivots calculated from previous day's OHLC
- Position sizing: 0.25
- Target: 20-40 trades/year (80-160 total over 4 years)
- Works in bull/bear: EMA50 defines trend, pivot breaks capture momentum, volume confirms
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_R1S1_Breakout_Volume_Trend_v2"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r1 = close + (range_ * 1.1 / 12)
    s1 = close - (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    s2 = close - (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    s3 = close - (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    s4 = close - (range_ * 1.1 / 2)
    return r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h: EMA50 trend filter ===
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily: Calculate Camarilla pivots from previous day's OHLC ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day (using previous day's data)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        r1, s1, _, _, _, _, _, _ = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Align daily pivots to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema50_val = ema50[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_val) or np.isnan(close_val) or np.isnan(vol_ratio_val) or 
            np.isnan(r1_val) or np.isnan(s1_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and uptrend
            if close_val > r1_val and vol_ratio_val > 1.5 and close_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and downtrend
            elif close_val < s1_val and vol_ratio_val > 1.5 and close_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA50 or touches S1
            if close_val < ema50_val or close_val <= s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above EMA50 or touches R1
            if close_val > ema50_val or close_val >= r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals