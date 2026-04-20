#!/usr/bin/env python3
"""
12h_1D_Pivot_R1S1_Breakout_Volume_Conservative_v1
Concept: 12h Camarilla pivot R1/S1 breakout with volume confirmation and 1d trend filter.
- Long: Close > R1 AND volume > 1.5x avg volume AND close > 1d EMA200
- Short: Close < S1 AND volume > 1.5x avg volume AND close < 1d EMA200
- Exit: Price crosses below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 12-37 trades/year (50-150 total over 4 years)
- Works in bull/bear: 1d EMA200 defines trend, Camarilla levels provide structure, volume confirms momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1D_Pivot_R1S1_Breakout_Volume_Conservative_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h: OHLC for Camarilla calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily: EMA200 trend filter ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === Daily: Volume context ===
    vol_1d = df_1d['volume'].values
    vol_ma50_1d = pd.Series(vol_1d).rolling(window=50, min_periods=50).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma50_1d > 0, vol_ma50_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        ema200_1d_val = ema200_1d_aligned[i]
        vol_ratio_1d_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_val) or np.isnan(low_val) or np.isnan(close_val) or 
            np.isnan(vol_ratio_val) or np.isnan(ema200_1d_val) or np.isnan(vol_ratio_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous bar's range
        if i > 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            if (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
                
            range_val = prev_high - prev_low
            if range_val <= 0:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            # Camarilla levels
            R1 = prev_close + (range_val * 1.1 / 12)
            S1 = prev_close - (range_val * 1.1 / 12)
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 AND volume confirmation AND above 1d EMA200
            vol_confirm = vol_ratio_val > 1.5 and vol_ratio_1d_val > 0.5  # Not extremely low volume
            
            if close_val > R1 and vol_confirm and close_val > ema200_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 AND volume confirmation AND below 1d EMA200
            elif close_val < S1 and vol_confirm and close_val < ema200_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below R1
            if close_val < R1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above S1
            if close_val > S1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals