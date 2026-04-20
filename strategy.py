#!/usr/bin/env python3
"""
6h_12h_1d_Pivot_R1S1_Breakout_MultiTimeframe_Confluence_v1
Concept: Multi-timeframe confluence of daily pivot points with 12h trend filter on 6h timeframe.
- Uses daily pivot points (R1, S1) from 1d as key support/resistance levels
- Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades
- Long when price breaks above R1 with volume confirmation AND 12h EMA50 trending up
- Short when price breaks below S1 with volume confirmation AND 12h EMA50 trending down
- Exit when price returns to central pivot point (mean reversion)
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: Pivot points adapt to market conditions, multi-timeframe filter reduces whipsaw
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Pivot_R1S1_Breakout_MultiTimeframe_Confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # === Calculate daily pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h: EMA50 trend filter ===
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema50_12h_val = ema50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_val) or np.isnan(pivot_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation AND 12h EMA50 trending up
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.3  # Volume above average
            uptrend = ema50_12h_val > ema50_12h[i-1] if i > 0 else True  # 12h EMA rising
            
            if breakout_long and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation AND 12h EMA50 trending down
            elif close_val < s1_val and vol_confirm and ema50_12h_val < ema50_12h[i-1] if i > 0 else False:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below central pivot (mean reversion)
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above central pivot (mean reversion)
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals