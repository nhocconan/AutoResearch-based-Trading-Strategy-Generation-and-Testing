#!/usr/bin/env python3
"""
1h_Pivot_R1S1_Breakout_Volume_Filter_v1
Concept: 1h Camarilla R1/S1 breakout with volume confirmation, filtered by 4h trend (EMA50) and 1d volume regime.
- Long: 1h close > R1 AND 4h close > EMA50 AND 1h volume > 1.5x 20-bar avg AND 1d volume ratio > 0.8
- Short: 1h close < S1 AND 4h close < EMA50 AND 1h volume > 1.5x 20-bar avg AND 1d volume ratio > 0.8
- Exit: Opposite breakout (S1 for longs, R1 for shorts) OR 4h trend flip
- Position sizing: 0.20
- Target: 15-37 trades/year (60-150 total over 4 years)
- Uses Camarilla pivots from prior day for intraday structure, volume for confirmation, multi-timeframe trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Pivot_R1S1_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # === Get 4h data ONCE before loop for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === Get 1d data ONCE before loop for volume regime and pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h: EMA50 trend filter ===
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d: Volume regime (avoid low volume days) ===
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma20_1d > 0, vol_ma20_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1d: Prior day OHLC for Camarilla pivot calculation (use previous day's data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day, then align to 1h
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_mult = 1.1 / 12
    r1_1d = close_1d + (high_1d - low_1d) * camarilla_mult
    s1_1d = close_1d - (high_1d - low_1d) * camarilla_mult
    
    # Align pivot levels to 1h timeframe (use prior day's levels for current day's trading)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1h: Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        close_val = prices['close'].iloc[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema50_4h_val = ema50_4h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_ratio_1d_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema50_4h_val) or 
            np.isnan(vol_ratio_val) or np.isnan(vol_ratio_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with 4h uptrend and volume confirmation
            if (close_val > r1_val and 
                ema50_4h_val > 0 and  # Valid EMA value
                close_val > ema50_4h_val and  # Price above 4h EMA50 (uptrend)
                vol_ratio_val > 1.5 and 
                vol_ratio_1d_val > 0.8):
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 with 4h downtrend and volume confirmation
            elif (close_val < s1_val and 
                  ema50_4h_val > 0 and  # Valid EMA value
                  close_val < ema50_4h_val and  # Price below 4h EMA50 (downtrend)
                  vol_ratio_val > 1.5 and 
                  vol_ratio_1d_val > 0.8):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Break below S1 (failed breakout) OR 4h trend flip to downtrend
            if close_val < s1_val or close_val < ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Break above R1 (failed breakdown) OR 4h trend flip to uptrend
            if close_val > r1_val or close_val > ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals