#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter + 1-day Camarilla pivot breakout with volume confirmation.
# Choppiness Index (CHOP) > 61.8 indicates ranging market (mean reversion), < 38.2 indicates trending.
# In ranging markets (CHOP > 61.8): Buy near S1/S2, Sell near R1/R2 pivots from daily timeframe.
# In trending markets (CHOP < 38.2): Breakout trades - Buy on break above R1, Sell on break below S1.
# Volume confirmation requires current volume > 1.5x 20-period average.
# This adapts to market regimes, reducing whipsaws in trends and capturing mean reversion in ranges.
# Target: 20-40 trades/year per symbol with controlled risk.
name = "4h_Choppiness_Camarilla_Pivot_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 14-period Choppiness Index
    # TR = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0)  # Default to neutral
    valid = (atr_14 > 0) & (max_hh > min_ll)
    chop[valid] = 100 * np.log10(atr_14[valid] * 14 / (max_hh[valid] - min_ll[valid])) / np.log10(14)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    pivot = (ph + pl + pc) / 3
    range_hl = ph - pl
    
    # Camarilla levels
    r4 = pc + range_hl * 1.5000
    r3 = pc + range_hl * 1.2500
    r2 = pc + range_hl * 1.1666
    r1 = pc + range_hl * 1.0833
    s1 = pc - range_hl * 1.0833
    s2 = pc - range_hl * 1.1666
    s3 = pc - range_hl * 1.2500
    s4 = pc - range_hl * 1.5000
    
    # Align to 4h timeframe (use previous day's levels)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for both indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop[i]) or np.isnan(r1_4h[i]) or np.isnan(r2_4h[i]) or 
            np.isnan(s1_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Determine market regime
            is_ranging = chop_val > 61.8
            is_trending = chop_val < 38.2
            
            if is_ranging:
                # Mean reversion in ranging market
                # Long near support, Short near resistance
                if close[i] <= s1_4h[i] and vol > 1.5 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= r1_4h[i] and vol > 1.5 * vol_ma:
                    signals[i] = -0.25
                    position = -1
            elif is_trending:
                # Breakout in trending market
                # Buy break above R1, Sell break below S1
                if close[i] > r1_4h[i] and vol > 1.5 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s1_4h[i] and vol > 1.5 * vol_ma:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if chop_val > 61.8 and close[i] >= r1_4h[i]:  # Take profit at resistance in range
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] < s1_4h[i]:  # Stop loss if breaks support in trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if chop_val > 61.8 and close[i] <= s1_4h[i]:  # Take profit at support in range
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] > r1_4h[i]:  # Stop loss if breaks resistance in trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals