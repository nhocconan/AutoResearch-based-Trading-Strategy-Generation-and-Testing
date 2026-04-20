#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Trend_v1
Concept: 4h Camarilla R1/S1 breakout with 1d trend filter and volume confirmation.
- Long: Close > R1 AND price > 1d EMA200 AND volume > 1.5x 20-period average
- Short: Close < S1 AND price < 1d EMA200 AND volume > 1.5x 20-period average
- Exit: Price crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 20-40 trades/year (80-160 total over 4 years)
- Uses proven Camarilla pivot levels from daily timeframe for institutional levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h: Price data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily: EMA200 trend filter ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === Daily: Camarilla levels (based on previous day's range) ===
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using prior day's close, high, low
    close_1d_shift = np.concatenate([[np.nan], close_1d[:-1]])
    high_1d_shift = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_shift = np.concatenate([[np.nan], low_1d[:-1]])
    
    r1_1d = close_1d_shift + (high_1d_shift - low_1d_shift) * 1.1 / 12
    s1_1d = close_1d_shift - (high_1d_shift - low_1d_shift) * 1.1 / 12
    
    # Align to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema200_val = ema200_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(ema200_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R1 AND price above 1d EMA200 AND volume confirmation
            if close_val > r1_val and close_val > ema200_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 AND price below 1d EMA200 AND volume confirmation
            elif close_val < s1_val and close_val < ema200_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals