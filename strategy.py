#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Regime
Concept: 4h Camarilla R1/S1 breakout with daily volume spike and choppiness regime filter.
- Long: Price > R1 (daily) AND daily volume > 1.5x 20-period avg AND Choppiness(14) < 40 (trending)
- Short: Price < S1 (daily) AND daily volume > 1.5x 20-period avg AND Choppiness(14) < 40 (trending)
- Exit: Price crosses back through Camarilla pivot point (PP)
- Position sizing: 0.25
- Target: 80-150 total trades over 4 years
- Works in bull/bear: Choppiness filter avoids ranging markets, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily: Camarilla Levels (Pivot Point, R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and support/resistance levels
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 12
    s1 = pp - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily Camarilla levels to 4h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_1d_vals = df_1d['volume'].values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
    
    # === 4h: Choppiness Index (14-period) for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: 100 * log10(sum(ATR)/ (max(high)-min(low)) ) / log10(14)
    # We calculate over 14-period window
    def calculate_chop(high, low, close, window=14):
        n = len(high)
        chop = np.full(n, np.nan)
        for i in range(window-1, n):
            # True Range sum over window
            tr_sum = 0
            for j in range(i-window+1, i+1):
                tr1 = high[j] - low[j]
                tr2 = abs(high[j] - close[j-1]) if j > 0 else abs(high[j] - close[j])
                tr3 = abs(low[j] - close[j-1]) if j > 0 else abs(low[j] - close[j])
                tr_sum += max(tr1, tr2, tr3)
            
            # Max high - min low over window
            max_high = np.max(high[i-window+1:i+1])
            min_low = np.min(low[i-window+1:i+1])
            range_hl = max_high - min_low
            
            if range_hl > 0 and tr_sum > 0:
                chop[i] = 100 * np.log10(tr_sum) / np.log10(window) / np.log10(range_hl) * np.log10(range_hl)
            else:
                chop[i] = 50  # neutral when invalid
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        current_vol = vol_1d_aligned[i]
        chop_val = chop[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ma) or np.isnan(current_vol) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-period average
        vol_condition = current_vol > 1.5 * vol_ma
        
        # Regime condition: Choppiness < 40 indicates trending market (not ranging)
        regime_condition = chop_val < 40
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and trending regime
            if close[i] > r1_val and vol_condition and regime_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and trending regime
            elif close[i] < s1_val and vol_condition and regime_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point (PP)
            if close[i] < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point (PP)
            if close[i] > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals